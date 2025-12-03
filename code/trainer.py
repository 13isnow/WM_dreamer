import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from tqdm import tqdm

class Trainer:
    def __init__(self, train_tokenizer_epochs=10, train_dynamics_epochs=10, train_agent_epochs=10):
        self.train_tokenizer_epochs = train_tokenizer_epochs
        self.train_dynamics_epochs = train_dynamics_epochs
        self.train_agent_epochs = train_agent_epochs

    def train_tokenizer(self, tokenizer, dataloader, optimizer, scheduler):
        tokenizer.train()

        lpips_loss_fn = nn.MSELoss() # lpips.LPIPS(net='alex')

        total_loss = 0.0
        for _ in tqdm(range(self.train_tokenizer_epochs), desc="Tokenizer Training Epochs"):
            for image_data, in dataloader:
                # 标准化
                B, T, H, W, C = image_data.shape
                image_data = image_data.reshape(B*T, C, H, W)
                image_data = tokenizer.preprocess(image_data)
                # 生成latent
                latent_tokens, mask = tokenizer(image_data)
                # 重建图像
                recon_image_data = tokenizer.decoder(latent_tokens)

                mse_loss = F.mse_loss(recon_image_data, image_data)
                lpips_loss = lpips_loss_fn(recon_image_data, image_data).mean()
                loss = mse_loss + 0.2 * lpips_loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
        return total_loss / self.train_tokenizer_epochs

    def train_dynamics(
            self, 
            dynamics, 
            tokenizer, 
            dataloader,
            tau_dim,
            tau_range,
            step_range,
            optimizer, 
            scheduler
        ):
        dynamics.train()
        total_loss = 0.0
        for _ in range(self.train_dynamics_epochs):
            epoch_loss = 0.0
            for image_data, actions in dataloader:
                # 生成latent
                B, T, C, H, W = image_data.shape
                image_data = image_data.reshape(B*T, C, H, W)
                image_data = tokenizer.preprocess(image_data)
                latent_tokens = tokenizer.encoder(image_data) # (B*T, L, token_dim)
                next_latent_tokens = torch.roll(latent_tokens, shifts=-1, dims=0)

                # 随机采样 tau 和 step
                tau = torch.rand(1).item() * (tau_range[1] - tau_range[0]) + tau_range[0]
                tau_idx = int((tau - tau_range[0]) / (tau_range[1] - tau_range[0]) * (tau_dim - 1))
                step_idx = torch.randint(0, len(step_range), (1,)).item()

                weight = 0.9 * tau + 0.1
                noise_tokens = dynamics.noise_generate(latent_tokens, tau)
                pred_next_latent_tokens = dynamics(noise_tokens, actions, tau_idx, step_idx)  # (B*T, L, token_dim)

                if step_idx == 0:
                    loss = F.mse_loss(pred_next_latent_tokens, next_latent_tokens)
                else:
                    with torch.no_grad():
                        halfway_pred = dynamics(noise_tokens, actions, tau_idx, step_idx - 1)
                        halfway_vector = (halfway_pred - latent_tokens) / (1 - tau)
                        halfway_latent = noise_tokens + halfway_vector * step_range[step_idx - 1]
                        halfway_tau = tau + step_range[step_idx - 1]
                        halfway_tau_idx = int((halfway_tau - tau_range[0]) / (tau_range[1] - tau_range[0]) * self.tau_dim)
                        end_pred = dynamics(halfway_latent, actions, halfway_tau_idx, step_idx - 1)
                        end_vector = (end_pred - halfway_pred) / (1 - halfway_tau)

                    pred_vector = (pred_next_latent_tokens - noise_tokens) / (1 - tau)
                    td_vector = (halfway_vector + end_vector).detach() / 2
                    loss = F.mse_loss(pred_vector, td_vector) * ((1 - tau) ** 2)

                epoch_loss += weight * loss

            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()

            total_loss += epoch_loss.item()

        scheduler.step()
        return total_loss / self.train_dynamics_epochs

    def calc_advantage(self, rewards, values, gamma, lambda_r):
        T = rewards.shape[1]
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[:, t + 1]
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * lambda_r * last_gae

        lambda_return = advantages + values
        return advantages, lambda_return

    def train_agent(
            self, 
            agent,
            dynamics,
            tokenizer,
            dataloader,
            task_ids,
            rewards,
            dynamics_optimizer,
            agent_optimizer,
            scheduler,
            tau_dim,
            tau_range,
            step_range,
            lambda_r,
            gamma
        ):
        agent.train()
        dynamics.train()
        tau_idx = (tau_dim - 1) # 使用最大tau
        tau = tau_idx * (tau_range[1] - tau_range[0]) + tau_range[0]
        step_idx = len(step_range) - 1 # 使用最大step
        
        total_loss = 0.0
        for _ in range(self.train_agent_epochs):
            for image_data, actions, task_ids, rewards in dataloader:
                # 奖励函数标准化
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                # 视频标准化
                B, T, C, H, W = image_data.shape
                image_data = image_data.reshape(B*T, C, H, W)
                image_data = tokenizer.preprocess(image_data)
                # Tokenizer 编码
                latent_tokens = tokenizer.encoder(image_data) # (B*T, L, token_dim)
                next_latent_tokens = torch.roll(latent_tokens, shifts=-1, dims=0)

                # Dynamics预测
                noise_tokens = dynamics.noise_generate(latent_tokens, tau)
                pred_latent_tokens = dynamics(noise_tokens, actions, tau_idx, step_idx)  # (B*T, L, token_dim)
                # 任务对齐
                aligned_latent = agent.task_aligning(pred_latent_tokens, task_ids)
                # 预测
                pred_action_logits, pred_values, pred_rewards = agent(aligned_latent)
                # 计算损失
                action_onehot = F.one_hot(actions.long(), num_classes=agent.action_dim).float()
                action_loss = F.cross_entropy(pred_action_logits, action_onehot)

                reward_loss = F.mse_loss(pred_rewards, rewards)

                _, lambda_return = self.calc_advantage(rewards, pred_values, gamma, lambda_r)
                value_loss = F.mse_loss(pred_values, lambda_return)

                agent_loss = action_loss + reward_loss + value_loss
                dynamics_loss = F.mse_loss(pred_latent_tokens, next_latent_tokens) * 0.9

                # Update
                dynamics_optimizer.zero_grad()
                dynamics_loss.backward()
                dynamics_optimizer.step()

                agent_optimizer.zero_grad()
                agent_loss.backward()
                agent_optimizer.step()
                total_loss += agent_loss.item()
        
            scheduler.step()
        return total_loss / self.train_agent_epochs

    def trajectory_generate(
            self, 
            agent, 
            dynamics, 
            latent_token,
            task_id,
            K_steps,
            tau_idx,
            step_idx
        ):
        latent_buffer = []
        value_buffer = []
        reward_buffer = []
        action_buffer = []

        for t in range(K_steps):
            aligned_latent = agent.task_aligning(latent_token, task_id)
            action_logits, value, reward = agent(aligned_latent)
            action, _ = agent.action_select(action_logits)
            with torch.no_grad():
                latent_token = dynamics(
                    latent_token, 
                    action, 
                    tau_idx=tau_idx, 
                    step_idx=step_idx
                )

            latent_buffer.append(latent_token)
            value_buffer.append(value)
            reward_buffer.append(reward)
            action_buffer.append(action)
        return latent_buffer, value_buffer, reward_buffer, action_buffer

    def calc_pmpo_loss(
            self,
            agent,
            prior_agent,
            latent_seqs,
            rewards,
            values,
            actions,
            task_id,
            gamma,
            lambda_r,
            alpha: float,
            beta: float,
        ):
        B, L, D = latent_seqs[0].shape
        K = len(rewards)
        
        seq_logits = []
        seq_dists = []
        seq_prior_logits = []
        seq_prior_dists = []
        for latent_token in latent_seqs:
            aligned_latent = agent.task_aligning(latent_token, task_id)
            action_logits, value, reward = agent(aligned_latent)
            with torch.no_grad():
                prior_action_logits, _, _ = prior_agent(aligned_latent)
            
            action, dist = agent.action_select(action_logits)
            prior_action, prior_dist = prior_agent.action_select(prior_action_logits)

            seq_logits.append(action_logits)
            seq_dists.append(dist)
            seq_prior_logits.append(prior_action_logits)
            seq_prior_dists.append(prior_dist)

        log_probs_list = []
        prior_log_probs_list = []
        kl_list = []
        for t in range(K):
            logits_t = seq_logits[t]
            dist_t = seq_dists[t]
            prior_dist_t = seq_prior_dists[t]
            action_t = actions[:, t]
            log_prob_t = dist_t.log_prob(action_t)
            
            probs = F.softmax(logits_t, dim=-1)
            probs_log = F.log_softmax(logits_t, dim=-1)
            prior_probs_log = F.log_softmax(seq_prior_logits[t], dim=-1)
            kl_t = (probs * (probs_log - prior_probs_log)).sum(dim=-1)
            
            log_probs_list.append(log_prob_t)
            prior_log_probs_list.append(prior_dist_t.log_prob(action_t))
            kl_list.append(kl_t)
        
        log_probs = torch.stack(log_probs_list, dim=1)          # (B, K)
        prior_log_probs = torch.stack(prior_log_probs_list, dim=1)  # (B, K)
        kl_divs = torch.stack(kl_list, dim=1)                  # (B, K)

        advantages, lambda_return = self.calc_advantage(rewards, values, gamma=gamma, lambda_r=lambda_r)

        pos_mask = (advantages >= 0.0)
        neg_mask = (advantages < 0.0)
        N_pos = pos_mask.sum().item()
        N_neg = neg_mask.sum().item()
        if N_pos > 0:
            loss_pos = - log_probs[pos_mask].mean()   # negative log-likelihood -> maximize prob
        else:
            loss_pos = torch.tensor(0.0)
        if N_neg > 0:
            loss_neg = log_probs[neg_mask].mean()    # positive mean logp -> penalize (reduce probability)
        else:
            loss_neg = torch.tensor(0.0)

        # 计算 Loss
        policy_loss = alpha * loss_pos + (1 - alpha) * loss_neg + beta * kl_divs.mean()
        value_loss = F.mse_loss(values, lambda_return)
        
        total_loss = policy_loss + value_loss
        return total_loss

    def train_pmpo(
            self,
            agent: nn.Module,
            dynamics: nn.Module,
            tokenizer: nn.Module,
            init_image_data: torch.Tensor,
            task_id: torch.Tensor,
            prior_agent: nn.Module,
            optimizer, 
            scheduler,
            tau_dim,
            tau_range,
            step_range,
            K_steps,
            lambda_r,
            gamma,
            alpha,
            beta
        ):
        agent.train()
        dynamics.eval()
        tokenizer.eval()
        prior_agent.eval()
        
        # dimensions
        B, T, C, H, W = init_image_data.shape
        L = tokenizer.latent_dim
        D = tokenizer.token_dim

        # Tokenizer 编码
        image_data = init_image_data.reshape(B*T, C, H, W)
        image_data = tokenizer.preprocess(image_data)
        latent_tokens = tokenizer.encoder(image_data) # (B*T, L, token_dim)
        # 提取 start latent
        start_latent = latent_tokens.reshape(B, T, L, D)[:, -1, :, :] # (B, L, D)
        # 轨迹生成参数
        tau_idx = (tau_dim - 1) # 使用最大tau
        step_idx = len(step_range) - 1 # 使用最大step
        imagined_latents, imagined_values, imagined_rewards, imagined_actions = self.trajectory_generate(
            agent,
            dynamics,
            start_latent,
            task_id,
            K_steps,
            tau_idx,
            step_idx
        )
        rewards = torch.stack(imagined_rewards, dim=1) # (B, K, 1)
        values = torch.stack(imagined_values, dim=1) # (B, K, 1)
        actions = torch.stack(imagined_actions, dim=1) # (B, K, 1)

        latent_seqs = [start_latent] + imagined_latents[:-1]
        pmpo_loss = self.calc_pmpo_loss(
            agent,
            prior_agent,
            latent_seqs,
            rewards.squeeze(-1),
            values.squeeze(-1),
            actions.squeeze(-1),
            task_id,
            gamma,
            lambda_r,
            alpha=alpha,
            beta=beta
        )

        optimizer.zero_grad()
        pmpo_loss.backward()
        optimizer.step()
        scheduler.step()
        return pmpo_loss.item()