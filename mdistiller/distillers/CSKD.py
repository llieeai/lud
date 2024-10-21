import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from ._base import Distiller
import numpy as np

def cosine_similarity(a, b, eps=1e-8):
    return torch.abs((a*b).sum(1)) / (a.norm(p=2, dim=1)*b.norm(p=2, dim=1) + eps)

def cosine_distance(stud, tea):
    return 1 - cosine_similarity(stud, tea).mean()

def cs_loss(logits_student, logits_teacher, temperature, alpha=2.0):
    pred_student = (logits_student / temperature).softmax(dim=1)
    pred_teacher = (logits_teacher / temperature).softmax(dim=1)
    cs_loss = alpha*temperature**2*cosine_distance(pred_student.transpose(0, 1), pred_teacher.transpose(0, 1))
    return cs_loss

def cc_loss(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)   
    consistency_loss = cosine_distance(teacher_matrix, student_matrix)
    return consistency_loss

def bc_loss(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    consistency_loss = cosine_distance(teacher_matrix, student_matrix)
    return consistency_loss

def cs_loss_mat(logits_student, logits_teacher, temperature, alpha=2.0):
    b_num, c_num = logits_student.shape
    T_mat = temperature.unsqueeze(1).repeat(1,c_num)

    pred_student = (logits_student / T_mat).softmax(dim=1)
    pred_teacher = (logits_teacher / T_mat).softmax(dim=1)
    cs_loss = alpha*temperature.mean()**2*cosine_distance(pred_student.transpose(0, 1), pred_teacher.transpose(0, 1))
    return cs_loss

def cc_loss_mat(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    T_mat = temperature.unsqueeze(1).repeat(1,class_num)

    pred_student = F.softmax(logits_student / T_mat, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_mat, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)   
    consistency_loss = cosine_distance(teacher_matrix, student_matrix)
    return consistency_loss

def bc_loss_mat(logits_student, logits_teacher, temperature):
    batch_size, class_num = logits_teacher.shape
    T_mat = temperature.unsqueeze(1).repeat(1,class_num)

    pred_student = F.softmax(logits_student / T_mat, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_mat, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    consistency_loss = cosine_distance(teacher_matrix, student_matrix)
    return consistency_loss

class CSKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(CSKD, self).__init__(student, teacher)
        self.temperature = cfg.CSKD.TEMPERATURE
        self.ce_loss_weight = cfg.CSKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.CSKD.LOSS.KD_WEIGHT
        
        self.low_temp = cfg.CSKD.CSWT.LOW_T
        self.high_temp = cfg.CSKD.CSWT.HIGH_T

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()
        
        #####################################################
        ### Cosine Similarity Weighted Temperature (CSWT) ###
        pred_student = F.softmax(logits_student_weak.detach(), dim=1)
        pred_teacher = F.softmax(logits_teacher_weak.detach(), dim=1)

        cs_value = cosine_similarity(pred_student, pred_teacher)
        cs_min = torch.min(cs_value).item()
        cs_max = torch.max(cs_value).item()

        T_mat = (self.high_temp - self.low_temp)/(cs_min - cs_max)*(cs_value - cs_max) + self.low_temp
        ### Cosine Similarity Weighted Temperature (CSWT) ###
        #####################################################

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_cs_weak = self.kd_loss_weight * ((cs_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((cs_loss_mat(
            logits_student_weak,
            logits_teacher_weak,
            T_mat,
        ) * mask).mean()) 
        
        loss_cs_strong = self.kd_loss_weight * cs_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) + self.kd_loss_weight * cs_loss_mat(
            logits_student_strong,
            logits_teacher_strong,
            T_mat,
        ) 

        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss_mat(
            logits_student_weak,
            logits_teacher_weak,
            T_mat,
        ) * class_conf_mask).mean()) 
        
        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss_mat(
            logits_student_weak,
            logits_teacher_weak,
            T_mat,
        ) * mask).mean()) 
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_cs_weak + loss_cs_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
        }

        return logits_student_weak, losses_dict


