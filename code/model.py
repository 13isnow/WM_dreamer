import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

def make_fc_layer(in_features: int, out_features: int):
    fc_layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)
    return fc_layer

class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list,
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
    
class Tokenizer(nn.Module):
    def __init__(
        self,
        image_dim: int,
        stride: int,
        kernel_size: int,
        padding: int,
        token_dim: int,
        num_tokens: int,
        mask_prob: float,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        # assert
        
        # args
        ## image
        self.image_h, self.image_w = image_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.patch_h = (self.image_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.patch_w = (self.image_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.num_patches = self.patch_h * self.patch_w
        ## token
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.mask_prob = mask_prob
        ## else
        self.num_heads = num_heads
        self.num_layers = num_layers
        # network
        self.normalize = transforms.Compose([
            transforms.RandomCrop((self.image_h, self.image_w)),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.conv_embed = nn.Sequential(
            nn.Conv2d(3, self.token_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.ReLU(),
        )
        self.tokenizer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.token_dim, 
                nhead=self.num_heads,
                batch_first=True
            ),
            num_layers=self.num_layers
        )
        self.reconv_embed = nn.Sequential(
            nn.ConvTranspose2d(self.token_dim, 3, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
        )
        # learned parameters
        self.latent_token = nn.Parameter(torch.randn(1, self.num_tokens, self.token_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.pos_token = nn.Parameter(torch.randn(1, self.num_patches + self.num_tokens, self.token_dim))

    def preprocess(self, image_data: torch.Tensor) -> torch.Tensor:
        # float
        image_data = image_data.float()
        # (B*T, 3, IMAGE_H, IMAGE_W) -> (B*T, token_dim, H, W)
        # [0, 255] -> [0, 1] -> [-1, 1]
        image_data = self.normalize(image_data / 255)
        return image_data

    def patch_generate(self, image_data):
        B = image_data.shape[0]
        # (B*T, 3, H, W) -> (B*T, token_dim, PH, PW) -> (B*T, PH*PW, token_dim)
        patch_tokens = self.conv_embed(image_data).permute(0, 2, 3, 1).reshape(B, self.num_patches, self.token_dim)
        return patch_tokens

    def latent_generate(self, patch_tokens):
        B = patch_tokens.shape[0]
        # (B*T, PH*PW + L, token_dim)
        joint_tokens = torch.cat([patch_tokens, self.latent_token.repeat(B, 1, 1)], dim=1)
        joint_tokens = joint_tokens + self.pos_token[:, : joint_tokens.shape[1], :]

        encoded_tokens = self.tokenizer(joint_tokens)
        # (B*T, PH*PW + L, token_dim) -> (B*T, L, token_dim)
        latent_tokens = encoded_tokens[:, -self.num_tokens:, :]
        return latent_tokens

    def apply_mask(self, x):
        B, L, D = x.shape # (b*t, num_patches, token_dim)
        mask = (torch.rand(B, L, device=x.device) < self.mask_prob).to(torch.bool)
        mask_token = self.mask_token.repeat(B, 1, 1)
        x_masked = torch.where(
            mask.unsqueeze(-1),  # (b*t, num_patches, 1)
            mask_token,    # 掩码位置填充
            x            # 非掩码位置保留原特征
        )
        return x_masked, mask

    def forward(self, image_data: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_generate(image_data)
        patch_tokens_masked, mask = self.apply_mask(patch_tokens)
        latent_tokens = self.latent_generate(patch_tokens_masked)
        return latent_tokens, mask

    @torch.no_grad()
    def encoder(self, image_data: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_generate(image_data)
        latent_tokens = self.latent_generate(patch_tokens)
        return latent_tokens

    def decoder(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        # (B*T, L, token_dim) -> (B*T, L + PH*PW, token_dim)
        B = latent_tokens.shape[0]
        pad_tokens = torch.zeros(B, self.num_patches, self.token_dim, device=latent_tokens.device)
        joint_tokens = torch.cat([latent_tokens, pad_tokens], dim=1)
        joint_tokens = joint_tokens + self.pos_token[:, : joint_tokens.shape[1], :]

        decoded_tokens = self.tokenizer(joint_tokens)
        # (B*T, PH*PW + L, token_dim) -> (B*T, PH*PW, token_dim) -> (B*T, PH, PW, token_dim) -> (B*T, token_dim, PH, PW)
        patch_tokens = decoded_tokens[:, -self.num_patches:, :].reshape(-1, self.patch_h, self.patch_w, self.token_dim).permute(0, 3, 1, 2)
        # (B*T, token_dim, PH, PW) -> (B*T, 3, H, W)
        image_data = self.reconv_embed(patch_tokens)
        return image_data

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
class Dynamics(nn.Module):
    def __init__(
            self,
            token_dim: int,
            action_dim: int,
            tau_dim: int,
            step_dim: int,
            num_registers: int,
            num_heads: int,
            num_layers: int,
        ):
        super().__init__()
        # args
        ## embedding
        self.action_dim = action_dim
        self.tau_dim = tau_dim
        self.step_dim = step_dim
        ## else
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # network
        ## embedding
        self.action_embed = nn.Embedding(self.action_dim, token_dim)
        self.tau_embed = nn.Embedding(self.tau_dim, token_dim)
        self.step_embed = nn.Embedding(self.step_dim, token_dim)
        ## transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.num_heads,
                batch_first=True
            ),
            num_layers=self.num_layers
        )

        # learned parameters
        self.register_tokens = nn.Parameter(torch.randn(1, num_registers, token_dim))
    
    def combine_generate(
            self, 
            latent_tokens: torch.Tensor, 
            action: torch.Tensor, 
            tau_idx: int, 
            step_idx: int
        ) -> torch.Tensor:
        B, L, D = latent_tokens.shape  # (b*t, l, token_dim)
        # embed action, tau, step
        action_embedded = self.action_embed(action).unsqueeze(1).repeat(1, L, 1)  # (b*t, l, token_dim)
        tau_embedded = self.tau_embed(tau_idx).unsqueeze(1).repeat(1, L, 1)  # (b*t, l, token_dim)
        step_embedded = self.step_embed(step_idx).unsqueeze(1).repeat(1, L, 1)  # (b*t, l, token_dim)
        # concatenate register tokens
        register_tokens = self.register_tokens.repeat(B, 1, 1)  # (b*t, num_registers, token_dim)
        combined = torch.cat([
            latent_tokens, 
            action_embedded,
            tau_embedded,
            step_embedded, 
            register_tokens
        ], dim=1)  # (b*t, 4*l + num_registers, token_dim)
        return combined

    def noise_generate(self, latent_tokens: torch.Tensor, tau: float) -> torch.Tensor:
        noise = torch.randn_like(latent_tokens)
        return tau * latent_tokens + (1 - tau) * noise

    def forward(self, noise_tokens, action, tau_idx, step_idx):
        B, L, D = noise_tokens.shape  # (b*t, l, token_dim)
        combined_tokens = self.combine_generate(noise_tokens, action, tau_idx, step_idx)
        pred_next_tokens = self.transformer(combined_tokens)
        pred_next_tokens = pred_next_tokens[:, -L:, :]  # (b*t, l, token_dim)
        return pred_next_tokens
    
class Agent(nn.Module):
    def __init__(
            self,
            token_dim: int,
            num_tasks: int,
            policy_layers: list,
            value_layers: list,
            reward_layers: list,
        ):
        super().__init__()
        # assert
        assert policy_layers[0] == token_dim, "Policy head input dimension must match token dimension."
        assert value_layers[0] == token_dim, "Value head input dimension must match token dimension."
        assert reward_layers[0] == token_dim, "Reward head input dimension must match token dimension."
        # args
        ## embedding
        self.num_tasks = num_tasks
        self.token_dim = token_dim
        ## else
        self.action_dim = policy_layers[-1]

        # network
        ## embedding
        self.task_embed = nn.Embedding(self.num_tasks, self.token_dim)
        ## heads
        self.policy_head = MLP(
            fc_feat_dim_list=policy_layers,
            name="policy_head",
            non_linearity=nn.ReLU,
            non_linearity_last=False,
        )
        self.value_head = MLP(
            fc_feat_dim_list=value_layers,
            name="value_head",
            non_linearity=nn.ReLU,
            non_linearity_last=False,
        )
        self.reward_head = MLP(
            fc_feat_dim_list=reward_layers,
            name="reward_head",
            non_linearity=nn.ReLU,
            non_linearity_last=False,
        )

    def task_aligning(self, latent_tokens, task_idx):
        B, L, D = latent_tokens.shape  # (b*t, l, token_dim)
        task_embedded = self.task_embed(task_idx).unsqueeze(1).repeat(1, L, 1)  # (b*t, l, token_dim)
        aligned_tokens = latent_tokens + task_embedded
        return aligned_tokens
    
    def forward(self, latent_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits = self.predict_action(latent_tokens) # (B*T, action_dim)
        value = self.predict_value(latent_tokens)   # (B*T, 1)
        reward = self.predict_reward(latent_tokens) # (B*T, 1)
        return action_logits, value, reward

    def predict_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(x)

    def predict_value(self, x: torch.Tensor) -> torch.Tensor: 
        return self.value_head(x)

    def predict_reward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reward_head(x)

    def action_select(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action_prob = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs=action_prob)
        if deterministic:
            action = torch.argmax(action_prob, dim=-1)
        else:
            action = action_dist.sample()
        return action, action_dist