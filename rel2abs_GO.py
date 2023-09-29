import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from evaluate import EvaluatePrompt


class Rel2abs_Decoder:
    def __init__(self, args, logger, target, src_anchors, anchors, tgt_distribution):
        self.absolute = args.absolute
        self.topk = args.topk

        self.budget = args.budget
        self.learning_rate = args.lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        target = torch.tensor(target).type(torch.float32).to(self.device)

        src_anchors = torch.tensor(src_anchors).type(torch.float32).to(self.device)
        self.target_rels = self.encode2rel(target, src_anchors)

        if self.topk > 0:
            self.target_rels, self.mask = self.zero_except_topk(self.target_rels)

        self.mean = torch.tensor(tgt_distribution[0]).to(self.device)
        self.std = torch.tensor(tgt_distribution[1]).to(self.device)

        self.anchors = torch.tensor(anchors).type(torch.float32).to(self.device)
        self.candidate = torch.empty((self.target_rels.shape[0], anchors.shape[1])).to(self.device)
        self.candidate.requires_grad = True
        torch.nn.init.xavier_normal_(self.candidate)

        self.target_abss = None

        self.eval_func = EvaluatePrompt(args)

        self.cos_loss = nn.CosineEmbeddingLoss()
        self.y = torch.ones(self.target_rels.shape[0]).to(self.device)

        self.logger = logger

        non_zero_indices = self.target_rels.nonzero(as_tuple=True)
        non_zero_values = self.target_rels[non_zero_indices]
        mean = non_zero_values.mean().item()
        std = non_zero_values.std().item()
        self.logger.info('Relative representations stat: mean %.4f, std %.4f' % (mean, std))
    
    def zero_except_topk(self, input_tensor):
        if self.absolute:
            _, topk_indices = torch.topk(torch.abs(input_tensor), self.topk)
        else:
            _, topk_indices = torch.topk(input_tensor, self.topk)

        mask = torch.zeros_like(input_tensor).to(self.device)
        mask.scatter_(-1, topk_indices, 1)
        masked_tensor = input_tensor * mask
        return masked_tensor, mask
    
    def regularize_tensor(self, tensor):
        current_mean = torch.mean(tensor).to(self.device)
        current_std = torch.std(tensor).to(self.device)
        
        normalized_tensor = (tensor - current_mean) / current_std
        regularized_tensor = normalized_tensor * self.std + self.mean
        
        return regularized_tensor

    def encode2rel(self, x, anchors):
        A = F.normalize(x, dim=-1)
        B = F.normalize(anchors, dim=-1)
        return torch.matmul(A, B.T)

    def set_target_abs(self, y):
        self.target_abss = torch.tensor(y).type(torch.float32)
        self.target_abss.requires_grad = False

    def eval(self, x):
        if self.target_abss == None:
            raise AssertionError('No target abs embedding defined?')
        cosine = nn.functional.cosine_similarity(x, self.target_abss, dim=-1)
        return torch.mean(cosine).item()

    def search(self):
        optimizer = optim.Adam([self.candidate], lr=self.learning_rate)

        best_precision = -1
        best_candidate = None

        pbar = tqdm(range(self.budget))
        for i in pbar:
            regularized_candidate = self.regularize_tensor(self.candidate)
            x_rel = self.encode2rel(regularized_candidate, self.anchors)
            if self.topk > 0:
                x_rel = x_rel * self.mask

            loss = self.cos_loss(x_rel, self.target_rels, self.y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (i+1) % 50 == 0:
                with torch.no_grad():
                    this_candidate = regularized_candidate.detach().cpu().numpy()
                    precision = self.eval_func.evaluate_valid(this_candidate)
                if precision > best_precision:
                    best_candidate = this_candidate
                    best_precision = precision
                    self.logger.info('Get best precision: %.4f at step %d! loss: %.4f' % (best_precision, i+1, loss.item()))
            pbar.set_description('best precision: %.4f, loss: %.4f' % (best_precision, loss.item()))

        with torch.no_grad():
            test_precision = self.eval_func.evaluate_test(best_candidate)
        self.logger.info('Test precision: %.4f' % test_precision)

        return best_candidate
