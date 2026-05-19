import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from model import Model_ggad, Model_ocgnn
from utils import *
import os
import dgl

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# Argument Configuration
# ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit')
parser.add_argument('--teacher_path', type=str, default='reddit_ggad_teacher_final.pth')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Set learning rate based on dataset
if args.lr is None:
    if args.dataset in ['Amazon']:
        args.lr = 5e-4
    elif args.dataset in ['tf_finace']:
        args.lr = 5e-4
    elif args.dataset in ['reddit']:
        args.lr = 5e-3
    elif args.dataset in ['elliptic']:
        args.lr = 1e-3
    elif args.dataset in ['photo']:
        args.lr = 5e-3
    elif args.dataset in ['tolokers']:
        args.lr = 5e-3
    elif args.dataset in ['YelpChi-all']:
        args.lr = 5e-4
    

# Set number of training epochs based on dataset
if args.num_epoch is None:
    if args.dataset in ['reddit']:
        args.num_epoch = 1000
    elif args.dataset in ['tf_finace']:
        args.num_epoch = 2500
    elif args.dataset in ['Amazon']:
        args.num_epoch = 1300
    elif args.dataset in ['elliptic']:
        args.num_epoch = 2000
    elif args.dataset in ['photo']:
        args.num_epoch = 2000
    elif args.dataset in ['tolokers']:
        args.num_epoch = 1500
    elif args.dataset in ['YelpChi-all']:
        args.num_epoch = 1500

# Noise parameter configuration
if args.dataset in ['reddit', 'Photo']:
    args.mean = 0.02
    args.var = 0.01
else:
    args.mean = 0.0
    args.var = 0.0

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

# ===========================
# Random Seed Configuration
# ===========================
print('Setting random seeds...')
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ===========================
# Helper Function Definitions
# ===========================

def distillation_loss_emb(emb_t, emb_s):
    """Compute distillation loss (MSE) between teacher and student embeddings."""
    loss = F.mse_loss(emb_s, emb_t, reduction='mean')
    return loss


def score_distillation_loss(score_t, score_s):
    """Compute distillation loss (MSE) between teacher and student scores."""
    loss = F.mse_loss(score_s, score_t, reduction='mean')
    return loss


def min_max_normalize(tensor):
    """Perform min-max normalization on tensor."""
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    if max_val == min_val:
        return torch.zeros_like(tensor)
    else:
        return (tensor - min_val) / (max_val - min_val)


def loss_func(emb):
    """OCGNN loss function for one-class objective."""
    r = 0
    beta = 0.5
    warmup = 2
    eps = 0.001
    c = torch.zeros(args.embedding_dim)
    dist = torch.sum(torch.pow(emb.cpu() - c, 2), 1)
    score = dist - r ** 2
    loss = r ** 2 + 1 / beta * torch.mean(torch.relu(score))

    if warmup > 0:
        with torch.no_grad():
            warmup -= 1
            r = torch.quantile(torch.sqrt(dist), 1 - beta)
            c = torch.mean(emb, 0)
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

    return loss, score, c, r


def kl_loss_student_teacher(student_score, teacher_score, eps=1e-8):
    """Compute KL divergence loss between student and teacher scores."""
    student_prob = torch.clamp(student_score, min=eps, max=1)
    teacher_prob = torch.clamp(teacher_score, min=eps, max=1)
    student_prob = student_prob / (student_prob.sum() + eps)
    teacher_prob = teacher_prob / (teacher_prob.sum() + eps)
    kl = F.kl_div(student_prob.log(), teacher_prob, reduction='batchmean')
    return kl





# ===========================
# Data Loading and Preprocessing
# ===========================
print('Loading and preprocessing data...')
adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_label_idx, abnormal_label_idx = load_mat(
    args.dataset)

# Preprocess features
if args.dataset in ['Amazon', 'tf_finace', 'reddit', 'elliptic','questions']:
    print('Preprocessing features...')
    features, _ = preprocess_features(features)
else:
    features = features.todense()

# Prepare input tensors
print('Preparing input tensors...')
nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
adj = normalize_adj(adj)

raw_adj = (raw_adj + sp.eye(raw_adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])

# ===========================
# Model Initialization
# ===========================
print('Initializing models...')

# Teacher model (GGAD) - Pretrained with frozen parameters
model = Model_ggad(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model.load_state_dict(torch.load(args.teacher_path))
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Student model (OCGNN) - To be trained
model_s = Model_ocgnn(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)

# Distillation MLPs
print('Initializing MLP for distillation...')
mlp_s = nn.Linear(args.embedding_dim, 1)  # Map embedding to anomaly score
pseudo_emb_mlp = nn.Linear(args.embedding_dim, args.embedding_dim)  # Pseudo anomaly embedding transformation

# ===========================
# Optimizer Initialization
# ===========================
print('Initializing optimizers...')
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_s = torch.optim.Adam(model_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_mlp_s = torch.optim.Adam(mlp_s.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimiser_pseudo_emb_mlp = torch.optim.Adam(pseudo_emb_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Loss functions
b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(
    [args.negsamp_ratio]).cuda() if torch.cuda.is_available() else torch.tensor([args.negsamp_ratio]))
xent = nn.CrossEntropyLoss()

# ========================================================
# Pre-compute Teacher Outputs (One-time Forward Pass)
# ========================================================
print("\n💡 Pre-computing teacher outputs (one-time forward pass)...")
log_message_initial = ""
with torch.no_grad():
    # 1. Get teacher model scores (for loss computation and evaluation)
    _, _, logits_total, _, _, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=False,
                                       )
    score_from_ggad_non_normalize = logits_total.squeeze(dim=-1).squeeze(0)  # [N] teacher anomaly scores

    # 2. Normalize teacher scores
    score_from_ggad = min_max_normalize(score_from_ggad_non_normalize)

    # 3. Compute and log teacher baseline AUC
    logits_teacher_test = np.squeeze(logits_total[:, idx_test, :].cpu().detach().numpy())
    auc_teacher = roc_auc_score(ano_label[idx_test], logits_teacher_test)
    log_message_initial = (f'Teacher baseline: Testing_last_ggad_ {args.dataset} AUC: {auc_teacher:.4f}\n')
    print(log_message_initial, end="")

    # 4. Get teacher embeddings and pseudo anomaly embeddings (for student model input)
    emb_t_all, _, _, _, emb_abnormal, _ = model(features, adj, abnormal_label_idx, normal_label_idx, train_flag=True)
    emb_t_all = emb_t_all.squeeze(0)  # [N, D] teacher embedding
    pseudo_emb = emb_abnormal.squeeze(0)  # [M, D] pseudo anomaly embedding

    # 5. Prepare concatenated scores for MSE loss
    pseudo_score_from_ggad = score_from_ggad[abnormal_label_idx]
    teacher_score_concat = torch.cat([score_from_ggad, pseudo_score_from_ggad], dim=0)

save_dir = f'./ggad_labeledNormal/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# ===========================
# Training Loop
# ===========================
print("\n🔁 Starting Student Training...")
output_file = f"./ggad_labeledNormal/{args.dataset}.txt"
with open(output_file, "a") as f:
    f.write(log_message_initial)  # Write teacher baseline AUC to file
    with tqdm(total=args.num_epoch) as pbar:
        total_time = 0
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            start_time = time.time()

            # Set training mode
            model_s.train()
            mlp_s.train()
            optimiser_s.zero_grad()
            optimiser_mlp_s.zero_grad()
            optimiser_pseudo_emb_mlp.zero_grad()

            # Transform pseudo anomaly embeddings (needs to be in loop since pseudo_emb_mlp is training)
            pseudo_emb_proj = pseudo_emb_mlp(pseudo_emb)  # [M, D]
            num_nodes = emb_t_all.size(0)

            # === Student model forward pass ===
            _, emb_s_all_raw = model_s(features, adj)
            emb_s_all = emb_s_all_raw.squeeze(0)  # [N, D] student embedding

            # Concatenate original nodes and pseudo anomaly node embeddings
            emb_concat = torch.cat([emb_s_all, pseudo_emb_proj], dim=0)  # [N+M, D]

            # Compute student anomaly scores
            student_score_non_normalize = mlp_s(emb_s_all).squeeze(dim=-1)  # [N] raw scores
            student_score = torch.sigmoid(student_score_non_normalize)  # [N] sigmoid scores

            # Compute concatenated anomaly scores
            student_score_concat_non_normalize = mlp_s(emb_concat).squeeze(dim=-1)  # [N+M]
            student_score_concat = min_max_normalize(student_score_concat_non_normalize)  # [N+M] normalized scores

            # === Compute second regularization term (reg2_mse) - Data augmentation ===

            # Step 1: Apply random mask augmentation to normal node input features
            all_normal_features = features.squeeze()[normal_label_idx]
            mask = torch.rand_like(all_normal_features) > 0.3  # 30% probability to mask (set to 0)
            masked_normal_features = all_normal_features * mask  # masked features

            # Step 2: Construct complete masked feature matrix
            masked_features = features.clone().squeeze()  # Copy original features
            # masked_features[all_normal_idx] = masked_normal_features  # Only replace normal node features
            masked_features[normal_label_idx] = masked_normal_features
            masked_features = masked_features.unsqueeze(0)  # Restore batch dimension [1, N, D]

            # Step 3: Input masked features to student network
            _, emb_s_augmented_raw = model_s(masked_features, adj)
            emb_s_augmented = emb_s_augmented_raw.squeeze(0)  # [N, D] augmented embedding

            # Step 4: Compute MSE loss between normal node embeddings before and after augmentation
            emb_s_normal_original = emb_s_all[normal_label_idx]
            emb_s_normal_augmented = emb_s_augmented[normal_label_idx]
            reg2_mse = F.mse_loss(emb_s_normal_augmented, emb_s_normal_original, reduction='mean')

            # === Compute MSE loss === (teacher_score_concat already computed outside loop)
            mse_loss = score_distillation_loss(student_score_concat, teacher_score_concat)

            # === Total loss computation ===
            total_loss = mse_loss + 0.01 * reg2_mse
            #                ^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^
            #                MSE loss        Second regularization term (data augmentation MSE)

            # === Backward pass and parameter update ===
            total_loss.backward()
            optimiser_s.step()
            optimiser_mlp_s.step()
            optimiser_pseudo_emb_mlp.step()

            # === Evaluation and logging (every 5 epochs) ===
            if epoch % 5 == 0:
                log_message = (
                    f"Epoch {epoch}: Total Loss = {total_loss.item()}\n"
                    f"MSE Loss = {mse_loss.item()}\n"
                    f"Reg2 MSE Loss = {reg2_mse.item()}\n"
                )

                # Switch to evaluation mode
                model_s.eval()

                # Log average scores
                log_message += f'student_score: {torch.mean(student_score)}\n'
                log_message += f'ggad_score: {torch.mean(score_from_ggad)}\n'

                # Extract test set scores
                logits_stu = np.squeeze(student_score[idx_test].cpu().detach().numpy())

                # Compute AUC
                auc_stu = roc_auc_score(ano_label[idx_test], logits_stu)

                # Log AUC results
                log_message += f'Testing {args.dataset} AUC_student_mlp_s: {auc_stu:.4f}\n'

                # Compute AP
                AP_stu = average_precision_score(ano_label[idx_test], logits_stu, average='macro', pos_label=1)

                # Log AP results
                log_message += f'Testing AP_student_mlp_s: {AP_stu:.4f}\n'
                log_message += f'Total time is: {total_time:.2f}\n'

                # Output and save logs
                print(log_message)
                f.write(log_message)
                f.flush()

            end_time = time.time()
            total_time += end_time - start_time