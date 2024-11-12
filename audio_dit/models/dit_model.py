import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.shape[1], :]
        return self.dropout(x)


def enc_dec_mask(T, S, frame_width=2, expansion=0, device='cuda'):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)


def pad_audio(audio, audio_unit=320, pad_threshold=80):
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')

    return audio


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, num_steps)
        elif mode == 'quadratic':
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == 'sigmoid':
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == 'cosine':
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f'Unknown diffusion schedule {mode}!')
        betas = torch.cat([torch.zeros(1), betas], dim=0)  # Padding beta_0 = 0

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.shape[0]):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.shape[0]):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.num_steps = num_steps
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = torch.randint(1, self.num_steps + 1, (batch_size,))
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffLiveHead(nn.Module):
    def __init__(self,
                 fps,
                 x_dim=63,
                 person_dim=63,
                 a_dim=768,
                 use_indicator=True,
                 feature_dim=512,
                 n_heads=8,
                 n_layers=8,
                 mlp_ratio=4.0,
                 align_mask_width=5,
                 use_learnable_pe=False,
                 use_shape_feat=False,
                 use_mouth_open_ratio=False,
                 n_motions = 65,
                 n_prev_motions = 10,
                 n_diff_steps = 50,
                 diff_schedule='cosine', # choices=['linear', 'cosine', 'quadratic', 'sigmoid']
                 cfg_mode = 'independent', # choices=['incremental', 'independent']
                 guiding_conditions = ['audio'], # choices=['audio', 'style']
                 device='cuda'):
        super().__init__()

        # Model parameters
        self.motion_feat_dim = x_dim # for live latent
        self.a_dim = a_dim
        self.hidden_dim = feature_dim
        self.person_dim = person_dim

        self.use_shape_feat = use_shape_feat
        self.use_mouth_open_ratio = use_mouth_open_ratio

        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        # if self.use_style:

        # Sampling parameters
        self.target = 'sample'

        # Audio encoder
        # if self.audio_model == 'wav2vec2':
        #     from .wav2vec2 import Wav2Vec2Model
        #     self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        #     # wav2vec 2.0 weights initialization
        #     self.audio_encoder.feature_extractor._freeze_parameters()
        # elif self.audio_model == 'hubert':
        #     from .hubert import HubertModel
        #     self.audio_encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960')
        #     self.audio_encoder.feature_extractor._freeze_parameters()

        #     frozen_layers = [0, 1]
        #     for name, param in self.audio_encoder.named_parameters():
        #         if name.startswith("feature_projection"):
        #             param.requires_grad = False
        #         if name.startswith("encoder.layers"):
        #             layer = int(name.split(".")[2])
        #             if layer in frozen_layers:
        #                 param.requires_grad = False
        # else:
        #     raise ValueError(f'Unknown audio model {self.audio_model}!')

        self.audio_feature_map = nn.Linear(self.a_dim, self.hidden_dim)
        self.start_audio_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, self.hidden_dim))

        self.start_motion_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, self.motion_feat_dim))

        # Diffusion model
        self.denoising_net = DenoisingNetwork(x_dim=self.motion_feat_dim,
                                              shape_dim=self.person_dim,
                                              use_indicator=use_indicator,
                                              feature_dim=self.hidden_dim,
                                              n_heads=n_heads,
                                              n_layers=n_layers,
                                              mlp_ratio=mlp_ratio,
                                              align_mask_width=align_mask_width,
                                              use_learnable_pe=use_learnable_pe,
                                              use_shape_feat=self.use_shape_feat,
                                              use_mouth_open_ratio=self.use_mouth_open_ratio,
                                              n_motions=n_motions,
                                              n_prev_motions=n_prev_motions,
                                              n_diff_steps=n_diff_steps,
                                              device=device)
        # diffusion schedule
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, mode=diff_schedule)

        # Classifier-free settings
        self.cfg_mode = cfg_mode
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['style', 'audio']]
        # if 'style' in self.guiding_conditions:
        #     if not self.use_style:
        #         raise ValueError('Cannot use style guiding without enabling it!')
        #     self.null_style_feat = nn.Parameter(torch.randn(1, 1, self.style_feat_dim))
        if 'audio' in self.guiding_conditions:
            audio_feat_dim = self.hidden_dim
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, audio_feat_dim))

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_or_feat, shape_feat=None, style_feat=None, mouth_open_ratio=None,
                prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_coef) motion coefficients or features
            audio_or_feat: (N, L_audio) raw audio or audio feature
            shape_feat: (N, 63), avg kp
            style_feat: (N, d_style)
            prev_motion_feat: (N, n_prev_motions, d_motion) previous motion coefficients or feature
            prev_audio_feat: (N, n_prev_motions, d_audio) previous audio features
            time_step: (N,)
            indicator: (N, L) 0/1 indicator of real (unpadded) motion coefficients

        Returns:
           motion_feat_noise: (N, L, d_motion)
        """
        # if self.use_style:
        #     assert style_feat is not None, 'Missing style features!'
        batch_size = motion_feat.shape[0]

        if self.use_mouth_open_ratio:
            assert mouth_open_ratio is not None, 'Missing mouth open ratio when mouth_open_ratio is enabled!'
        if self.use_shape_feat:
            assert shape_feat is not None, 'Missing shape features when shape_feat is enabled!'

        # if audio_or_feat.ndim == 2:
        #     # Extract audio features
        #     assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
        #         f'Incorrect audio length {audio_or_feat.shape[1]}'
        #     audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        # elif audio_or_feat.ndim == 3:
        #     assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
        #     audio_feat_saved = audio_or_feat
        # else:
        #     raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        # audio_feat = audio_feat_saved.clone()
        prev_audio_feat = self.audio_feature_map(prev_audio_feat)  # (N, L_p, feature_dim)
        audio_feat = self.audio_feature_map(audio_or_feat)  # (N, L_p + L, feature_dim)

        # if shape_feat.ndim == 2:
        #     shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)
        # if style_feat is not None and style_feat.ndim == 2:
        #     style_feat = style_feat.unsqueeze(1)  # (N, 1, d_style)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        # Classifier-free guidance
        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                # if 'style' in self.guiding_conditions:
                #     mask_style = torch.rand(batch_size, device=self.device) < null_cond_prob
                #     style_feat = torch.where(mask_style.view(-1, 1, 1),
                #                              self.null_style_feat.expand(batch_size, -1, -1),
                #                              style_feat)
                if 'audio' in self.guiding_conditions:
                    mask_audio = torch.rand(batch_size, device=self.device) < null_cond_prob
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)
            else:
                # len(self.guiding_conditions) > 1 and self.cfg_mode == 'incremental'
                # full (0.45), w/o style (0.45), w/o style or audio (0.1)
                mask_flag = torch.rand(batch_size, device=self.device)
                # if 'style' in self.guiding_conditions:
                #     mask_style = mask_flag > 0.55
                #     style_feat = torch.where(mask_style.view(-1, 1, 1),
                #                              self.null_style_feat.expand(batch_size, -1, -1),
                #                              style_feat)
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag > 0.9
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)

        # if style_feat is None and shape_feat is not None:
        #     # The model only accepts audio and shape features, i.e., self.use_style = False
        #     person_feat = shape_feat
        # else:
        #     person_feat = torch.cat([shape_feat, style_feat], dim=-1)

        if time_step is None:
            # Sample time step
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # (N,)

        # The forward diffusion process
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # (N,)
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (N, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (N, 1, 1)

        eps = torch.randn_like(motion_feat)  # (N, L, d_motion)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        # The reverse diffusion process
        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat,
                                                prev_motion_feat, prev_audio_feat, time_step, indicator,
                                                shape_feat, mouth_open_ratio)

        return eps, motion_feat_target, motion_feat.detach(), audio_feat.detach()

    # def extract_audio_feature(self, audio, frame_num=None):
    #     frame_num = frame_num or self.n_motions

    #     # # Strategy 1: resample during audio feature extraction
    #     # hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num).last_hidden_state  # (N, L, 768)

    #     # Strategy 2: resample after audio feature extraction (BackResample)
    #     hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
    #                                        frame_num=frame_num * 2).last_hidden_state  # (N, 2L, 768)
    #     hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
    #     hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')  # (N, 768, L)
    #     hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)

    #     audio_feat = self.audio_feature_map(hidden_states)  # (N, L, feature_dim)
    #     return audio_feat

    @torch.no_grad()
    def sample(self, audio_or_feat, shape_feat, style_feat=None, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False):
        # Check and convert inputs
        batch_size = audio_or_feat.shape[0]

        # Check CFG conditions
        if cfg_mode is None:  # Use default CFG mode
            cfg_mode = self.cfg_mode
        if cfg_cond is None:  # Use default CFG conditions
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'style']]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        # sort cfg_cond and cfg_scale
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'style'].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        if 'style' in cfg_cond:
            assert self.use_style and style_feat is not None

        if self.use_style:
            if style_feat is None:  # use null style feature
                style_feat = self.null_style_feat.expand(batch_size, -1, -1)
        else:
            assert style_feat is None, 'This model does not support style feature input!'

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        # if shape_feat.ndim == 2:
        #     shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)
        # if style_feat is not None and style_feat.ndim == 2:
        #     style_feat = style_feat.unsqueeze(1)  # (N, 1, d_style)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        # Prepare input for the reverse diffusion process (including optional classifier-free guidance)
        if 'audio' in cfg_cond:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        if 'style' in cfg_cond:
            person_feat_null = torch.cat([shape_feat, self.null_style_feat.expand(batch_size, -1, -1)], dim=-1)
        else:
            if self.use_style:
                person_feat_null = torch.cat([shape_feat, style_feat], dim=-1)
            else:
                person_feat_null = shape_feat

        audio_feat_in = [audio_feat_null]
        # person_feat_in = [person_feat_null]
        for cond in cfg_cond:
            if cond == 'audio':
                audio_feat_in.append(audio_feat)
                # person_feat_in.append(person_feat_null)
            elif cond == 'style':
                if cfg_mode == 'independent':
                    audio_feat_in.append(audio_feat_null)
                elif cfg_mode == 'incremental':
                    audio_feat_in.append(audio_feat)
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')
                # person_feat_in.append(torch.cat([shape_feat, style_feat], dim=-1))

        n_entries = len(audio_feat_in)
        audio_feat_in = torch.cat(audio_feat_in, dim=0)
        # person_feat_in = torch.cat(person_feat_in, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            if t > 1:
                z = torch.randn_like(motion_at_T)
            else:
                z = torch.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(motion_in, audio_feat_in, None, # person_feat_in is not used
                                         prev_motion_feat_in, prev_audio_feat_in, step_in, indicator_in)

            # Apply thresholding if specified
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)

            # ( 1 + cfg ) * Gen(C) - cfg * Gen(C0)
            target_theta = results[1][:, -self.n_motions:]
            # Classifier-free Guidance (optional)
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta += cfg_scale[i] * (
                                 - results[0][:, -self.n_motions:])
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:])
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))

            traj[t - 1] = motion_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        else:
            return traj[0], motion_at_T, audio_feat

    @torch.no_grad()
    def sample_new(self, audio_or_feat, shape_feat, style_feat=None, mouth_open_ratio=None,
                   prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, total_denoising_steps=None, cfg_mode=None,
            #     indicator=None,
               cfg_cond=None, cfg_scale=1, flexibility=0,
               dynamic_threshold=None, ret_traj=False):
        # Check and convert inputs
        batch_size = audio_or_feat.shape[0]

        if self.use_mouth_open_ratio:
            assert mouth_open_ratio is not None, 'Missing mouth open ratio when mouth_open_ratio is enabled!'
        if self.use_shape_feat:
            assert shape_feat is not None, 'Missing shape features when shape_feat is enabled!'

        audio_or_feat = self.audio_feature_map(audio_or_feat)
        prev_audio_feat = self.audio_feature_map(prev_audio_feat)

        # Ensure shape_feat is 3D
        # if shape_feat.ndim == 2:
        #     shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)

        # Check CFG conditions
        if cfg_mode is None:  # Use default CFG mode
            cfg_mode = self.cfg_mode
        if cfg_cond is None:  # Use default CFG conditions
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'style']]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        # sort cfg_cond and cfg_scale
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'style'].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        # if 'style' in cfg_cond:
        #     assert self.use_style and style_feat is not None

        # if self.use_style:
        #     if style_feat is None:  # use null style feature
        #         style_feat = self.null_style_feat.expand(batch_size, -1, -1)
        # else:
        #     assert style_feat is None, 'This model does not support style feature input!'

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        # if shape_feat.ndim == 2:
        #     shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)
        # if style_feat is not None and style_feat.ndim == 2:
        #     style_feat = style_feat.unsqueeze(1)  # (N, 1, d_style)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        # Prepare input for the reverse diffusion process (including optional classifier-free guidance)
        if 'audio' in cfg_cond:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        # if 'style' in cfg_cond:
        #     person_feat_null = torch.cat([shape_feat, self.null_style_feat.expand(batch_size, -1, -1)], dim=-1)
        # else:
        #     if self.use_style:
        #         person_feat_null = torch.cat([shape_feat, style_feat], dim=-1)
        #     else:
        person_feat_null = shape_feat

        audio_feat_in = [audio_feat_null]
        person_feat_in = [person_feat_null]
        for cond in cfg_cond:
            if cond == 'audio':
                audio_feat_in.append(audio_feat)
                person_feat_in.append(person_feat_null)
            # elif cond == 'style':
            #     if cfg_mode == 'independent':
            #         audio_feat_in.append(audio_feat_null)
            #     elif cfg_mode == 'incremental':
            #         audio_feat_in.append(audio_feat)
            #     else:
            #         raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')
                # person_feat_in.append(torch.cat([shape_feat, style_feat], dim=-1))

        n_entries = len(audio_feat_in)
        audio_feat_in = torch.cat(audio_feat_in, dim=0)
        person_feat_in = torch.cat(person_feat_in, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        # indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        if total_denoising_steps is None:
            total_denoising_steps = self.diffusion_sched.num_steps
        traj = {total_denoising_steps: motion_at_T}
        use_og_diffusion = True
        for t in range(total_denoising_steps, 0, -1):
            if t > 1:
                z = torch.randn_like(motion_at_T)
            else:
                z = torch.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(
                motion_feat = motion_in, audio_feat = audio_feat_in,
                prev_motion_feat = prev_motion_feat_in, prev_audio_feat = prev_audio_feat_in,
                step = step_in, indicator=None, shape_feat=shape_feat, mouth_open_ratio=mouth_open_ratio)

            # Apply thresholding if specified
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)

            # Unconditional target (CFG) or the conditional target (non-CFG)
            if use_og_diffusion:
                target_theta = results[0][:, -self.n_motions:]
            else:
                target_theta = results[1][:, -self.n_motions:]
            # Classifier-free Guidance (optional)
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    if use_og_diffusion:
                        target_theta += cfg_scale[i] * (
                                    results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])
                    else:
                        target_theta += cfg_scale[i] * (
                                 - results[0][:, -self.n_motions:])
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:])
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
                # print(f"Step {t}, motion_next: {motion_next[0:1, 20:23, -6:]},\
                #       motion_at_t: {motion_at_t[0:1, 20:23, -6:]},\
                #       target_theta: {target_theta[0:1, 20:23, -6:]},\
                #       z: {z[0:1, 20:23, -6:]}, \n c0: {c0}, c1: {c1}, sigma: {sigma}, alpha: {alpha}, alpha_bar: {alpha_bar}, alpha_bar_prev: {alpha_bar_prev}")
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))

            traj[t - 1] = motion_next.detach()  # Stop gradient and save trajectory.
            # print(f'traj[t - 1]: last 6: {traj[t - 1][0:1, 20:23, -6:]}')
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        # if ret_traj:
        #     return traj, motion_at_T, audio_feat
        # else:
        #     return traj[0], motion_at_T, audio_feat
        return traj[0]

class DenoisingNetwork(nn.Module):
    def __init__(self,
                 x_dim,
                 shape_dim,
                 use_indicator=True,
                 feature_dim=512,
                 n_heads=8,
                 n_layers=8,
                 mlp_ratio=4.0,
                 align_mask_width=1,
                 use_learnable_pe=False,
                 use_mouth_open_ratio=False,
                 use_shape_feat=False,
                 n_motions = 65,
                 n_prev_motions = 10,
                 n_diff_steps = 50,
                 device='cuda'):
        super().__init__()

        # Model parameters
        self.motion_feat_dim = x_dim
        self.shape_feat_dim = shape_dim

        self.use_indicator = use_indicator
        self.use_learnable_pe = use_learnable_pe
        self.use_shape_feat = use_shape_feat
        self.use_mouth_open_ratio = use_mouth_open_ratio
        # Transformer
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        # sequence length
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        # Temporal embedding for the diffusion time step
        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        if self.use_learnable_pe:
            # Learnable positional encoding
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        if use_shape_feat:
            self.shape_proj = nn.Linear(self.shape_feat_dim, self.feature_dim)
        if use_mouth_open_ratio:
            self.mouth_open_ratio_proj = nn.Linear(1, self.feature_dim)

        # Transformer decoder
        self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0),
                                        self.feature_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward= 4 * self.feature_dim,
            activation='gelu', batch_first=True, dtype=torch.float32, device=device)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
        if self.align_mask_width > 0:
            motion_len = self.n_prev_motions + self.n_motions
            alignment_mask = enc_dec_mask(motion_len, motion_len, 1, self.align_mask_width - 1)
            alignment_mask = F.pad(alignment_mask, (0, 0, 1, 0), value=False)
            self.register_buffer('alignment_mask', alignment_mask)
        else:
            self.alignment_mask = None

        # Motion decoder
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None, shape_feat=None, mouth_open_ratio=None):
        """
        Args:
            motion_feat: (N, L, d_motion). Noisy motion feature
            audio_feat: (N, L, feature_dim)
            shape_feat: (N, 1, d_shape)
            prev_motion_feat: (N, L_p, d_motion). Padded previous motion coefficients or feature
            prev_audio_feat: (N, L_p, d_audio). Padded previous motion coefficients or feature
            step: (N,)
            indicator: (N, L). 0/1 indicator for the real (unpadded) motion feature

        Returns:
            motion_feat_target: (N, L_p + L, d_motion)
        """
        # Diffusion time step embedding
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)  # (N, 1, diff_step_dim)
        cond_feat = diff_step_embedding
        if self.use_shape_feat:
            shape_feat = self.shape_proj(shape_feat).unsqueeze(1)  # (N, 1, feature_dim)
            cond_feat += shape_feat
        if self.use_mouth_open_ratio:
            mouth_open_feat = self.mouth_open_ratio_proj(mouth_open_ratio).unsqueeze(1)  # (N, 1, feature_dim)
            cond_feat += mouth_open_feat

        if indicator is not None:
            indicator = torch.cat([torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                                   indicator], dim=1)  # (N, L_p + L)
            indicator = indicator.unsqueeze(-1)  # (N, L_p + L, 1)

        # Concat features and embeddings
        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)  # (N, L_p + L, d_motion)

        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)  # (N, L_p + L, d_motion + d_audio + 1)

        feats_in = self.feature_proj(feats_in)  # (N, L_p + L, feature_dim)
        feats_in = torch.cat([cond_feat, feats_in], dim=1)  # (N, 1 + L_p + L, feature_dim)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        # Transformer
        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)  # (N, L_p + L, d_audio)
        feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)

        # Decode predicted motion feature noise / sample
        motion_feat_target = self.motion_dec(feat_out[:, 1:])  # (N, L_p + L, d_motion)

        return motion_feat_target
