import numpy as np
import torch
import torch.nn as nn

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from gradcam import ModelOutputs


class SBSM(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(SBSM, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, window_size, stride, savepath='masks.npy'):
        """
        Generates sliding window type binary masks used in augment() to 
        mask an image. The Images are resized to 224x224 to 
        enable re-use of masks Generating the sliding window style masks.
        :param int window_size: the block window size 
        (with value 0, other areas with value 1)
        :param int stride: the sliding step
        :param tuple image_size: the mask size which should be the 
        same to the image size
        :return: the sliding window style masks
        :rtype: numpy.ndarray
        """

        rows = np.arange(0 + stride - window_size, self.input_size[0], stride)
        cols = np.arange(0 + stride - window_size, self.input_size[1], stride)

        mask_num = len(rows) * len(cols)
        masks = np.ones(
            (mask_num, self.input_size[0], self.input_size[1]), dtype=np.uint8)
        i = 0
        for r in rows:
            for c in cols:
                if r < 0:
                    r1 = 0
                else:
                    r1 = r
                if r + window_size > self.input_size[0]:
                    r2 = self.input_size[0]
                else:
                    r2 = r + window_size
                if c < 0:
                    c1 = 0
                else:
                    c1 = c
                if c + window_size > self.input_size[1]:
                    c2 = self.input_size[1]
                else:
                    c2 = c + window_size
                masks[i, r1:r2, c1:c2] = 0
                i += 1
        masks = masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, masks)
        self.register_buffer('masks', torch.from_numpy(masks).cuda())
        self.N = self.masks.shape[0]
        self.window_size = window_size
        self.stride = stride

    def load_masks(self, filepath):
        masks = np.load(filepath)
        self.register_buffer('masks', torch.from_numpy(masks).cuda())
        self.N = self.masks.shape[0]

    def weighted_avg(self, K):
        count = self.N - self.masks.sum(dim=(0, 1))
        sal = K.sum(dim=-1).permute(2, 0, 1) / count

        return sal

    def forward(self, x_q, x):
        with torch.no_grad():
            # Get embedding of query and retrieval image
            x_q = self.model(x_q)
            x_r = self.model(x)
            o_dist = torch.cdist(x_q, x_r)

            # Apply array of masks to the image
            stack = torch.mul(self.masks, x)

            x = []
            for i in range(0, self.N, self.gpu_batch):
                x.append(self.model(stack[i:min(i + self.gpu_batch, self.N)]))
            x = torch.cat(x)
            m_dist = torch.cdist(x_q, x)

            # Compute saliency
            K = (1 - self.masks).permute(2, 3, 1, 0) * \
                (m_dist - o_dist).clamp(min=0)
            sal = self.weighted_avg(K)

        return sal


class SBSMBatch(SBSM):
    def forward(self, x_q, x=None):
        if x is None:
            x = x_q
            self_sim = True
        else:
            self_sim = False
        B, C, H, W = x.size()

        with torch.no_grad():
            # Get embedding of query and retrieval images
            x_q = self.model(x_q)
            if not self_sim:
                x_r = self.model(x)
                o_dist = torch.cdist(x_q, x_r)
                o_dist = o_dist.view(-1, 1)

            # Apply array of masks to the image
            stack = torch.mul(self.masks.view(self.N, 1, H, W),
                              x.view(B * C, H, W))
            stack = stack.view(B * self.N, C, H, W)

            x = []
            for i in range(0, self.N*B, self.gpu_batch):
                x.append(self.model(
                    stack[i:min(i + self.gpu_batch, self.N*B)]))
            x = torch.cat(x)
            if self_sim:
                m_dist = torch.norm(x_q.unsqueeze(
                    1) - x.view(-1, B, x_q.shape[1]).permute(1, 0, 2), dim=2)
            else:
                m_dist = torch.cdist(x_q, x)

            # Compute saliency
            if self_sim:
                K = (1 - self.masks).permute(2, 3, 1, 0) * m_dist
            else:
                m_dist = m_dist.view(-1, self.N, B).permute(0,
                                                            2, 1).reshape(-1, self.N)
                K = (1 - self.masks).permute(2, 3, 1, 0) * \
                    (m_dist - o_dist).clamp(min=0)
            sal = self.weighted_avg(K)

        return sal


class SBSMMask(SBSM):
    def __init__(self, model, input_size, mode='mask', sigma=8, gpu_batch=100):
        super(SBSMMask, self).__init__(model, input_size, gpu_batch)
        self.mode = mode
        if self.mode == 'blur':
            self.kernel = self.create_kernel(sigma)

    # Source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    def create_kernel(self, sigma, channels=3, dim=2):
        import math
        import numbers
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        # Define kernel size based on sigma
        width = [math.ceil(4 * s) for s in sigma]
        kernel_size = [len(range(-w, w+1)) for w in width]

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        return kernel

    def blur(self, x):
        return nn.functional.conv2d(x, self.kernel.to(x.device),
                                    padding=(self.kernel.shape[2]//2,
                                             self.kernel.shape[3]//2),
                                    groups=x.shape[1])

    def norm(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean, device=x.device)[:, None, None]
        std = torch.tensor(std, device=x.device)[:, None, None]

        return (x - mean) / std

    def denorm(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean, device=x.device)[:, None, None]
        std = torch.tensor(std, device=x.device)[:, None, None]

        return (x * std) + mean

    def transform(self, x, mask):
        """
        Mask should be a tensor of shape Bx1xHxW
        """
        assert (x.shape[0], x.shape[2], x.shape[3]) == (
            mask.shape[0], mask.shape[2], mask.shape[3])

        if self.mode == 'mask':
            return mask * x
        elif self.mode == 'blur':
            x_blur = self.norm(self.blur(self.denorm(x)))
            return mask * x + (1 - mask) * x_blur
        else:
            print('Unsupported mode!')

    def forward(self, x_q, x, mask):
        with torch.no_grad():
            # Mask/blur image if needed
            if self.mode is not None:
                x_q = self.transform(x_q, mask)
            # Get embedding of query and retrieval image
            x_q = self.model(x_q)
            x_r = self.model(x)
            o_dist = torch.cdist(x_q, x_r)

            # Apply array of masks to the image
            stack = torch.mul(self.masks, x)

            x = []
            for i in range(0, self.N, self.gpu_batch):
                x.append(self.model(stack[i:min(i + self.gpu_batch, self.N)]))
            x = torch.cat(x)
            m_dist = torch.cdist(x_q, x)

            # Compute saliency
            K = (1 - self.masks).permute(2, 3, 1, 0) * \
                (m_dist - o_dist).clamp(min=0)
            sal = self.weighted_avg(K)

        return sal


class SBSMFeatureMask(SBSM):
    def __init__(self, model, input_size, feature_module, target_layers, gpu_batch=100):
        super(SBSMFeatureMask, self).__init__(model, input_size, gpu_batch)
        self.feature_module = feature_module

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layers, return_gradients=False)

    def forward(self, x_q, x, mask):
        with torch.no_grad():
            # Get embedding of retrieval image
            x_r = self.model(x)

            # Extract intermediate activations and embedding of query image
            A, _ = self.extractor(x_q)

            # Apply array of masks to the last conv layer
            feats = A[-1]  # choose last set of features
            feats = nn.functional.interpolate(
                feats, size=self.input_size, mode='bilinear')
            feats = feats * mask
            feats = feats.sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
            o_dist = torch.cdist(feats, x_r)

            # Apply array of masks to the image
            stack = torch.mul(self.masks, x)

            x = []
            for i in range(0, self.N, self.gpu_batch):
                x.append(self.model(stack[i:min(i + self.gpu_batch, self.N)]))
            x = torch.cat(x)
            m_dist = torch.cdist(feats, x)

            # Compute saliency
            K = (1 - self.masks).permute(2, 3, 1, 0) * \
                (m_dist - o_dist).clamp(min=0)
            sal = self.weighted_avg(K)

        return sal


class SBSMFeature(nn.Module):
    def __init__(self, model, feature_module, target_layers):
        super(SBSMFeature, self).__init__()
        self.model = model
        self.feature_module = feature_module

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layers, return_gradients=False)

    def generate_masks(self, input_size, window_size=1, stride=1):
        """
        Generates sliding window type binary masks used in augment() to 
        mask an image. The Images are resized to 224x224 to 
        enable re-use of masks Generating the sliding window style masks.
        :param int window_size: the block window size 
        (with value 0, other areas with value 1)
        :param int stride: the sliding step
        :param tuple image_size: the mask size which should be the 
        same to the image size
        :return: the sliding window style masks
        :rtype: numpy.ndarray
        """

        rows = np.arange(0 + stride - window_size, input_size[0], stride)
        cols = np.arange(0 + stride - window_size, input_size[1], stride)

        mask_num = len(rows) * len(cols)
        masks = np.ones(
            (mask_num, input_size[0], input_size[1]), dtype=np.uint8)
        i = 0
        for r in rows:
            for c in cols:
                if r < 0:
                    r1 = 0
                else:
                    r1 = r
                if r + window_size > input_size[0]:
                    r2 = input_size[0]
                else:
                    r2 = r + window_size
                if c < 0:
                    c1 = 0
                else:
                    c1 = c
                if c + window_size > input_size[1]:
                    c2 = input_size[1]
                else:
                    c2 = c + window_size
                masks[i, r1:r2, c1:c2] = 0
                i += 1
        masks = masks.reshape(-1, 1, *input_size)
        self.register_buffer('masks', torch.from_numpy(masks).cuda())
        self.N = self.masks.shape[0]
        self.window_size = window_size
        self.stride = stride

    def weighted_avg(self, K):
        count = self.N - self.masks.sum(dim=(0, 1))
        sal = K.sum(dim=-1).permute(2, 0, 1) / count

        return sal

    def forward(self, x_q, x):
        _, _, H, W = x_q.size()

        with torch.no_grad():
            # Get embedding of retrieval image
            x_r = self.model(x)

            # Extract intermediate activations and embedding of query image
            A, x_q = self.extractor(x_q)
            o_dist = torch.cdist(x_q, x_r)

            # Apply array of masks to the last conv layer
            feats = A[-1]  # choose last set of features
            stack = torch.mul(self.masks, feats[1])

            # Assume stride 1 here
            stack = stack.sum(dim=(2, 3)) / \
                (self.masks.shape[2] * self.masks.shape[3] - 1)
            m_dist = torch.cdist(x[0].unsqueeze(0), stack)

            # Compute saliency
            K = (1 - self.masks).permute(2, 3, 1, 0) * \
                (m_dist - o_dist).clamp(min=0)
            sal = self.weighted_avg(K)

            # Upsample
            sal = nn.functional.interpolate(sal.unsqueeze(
                1), size=(H, W), mode='bilinear').squeeze(1)

        return sal


class FSal(SBSM):
    def get_classifier(self, classifier):
        self.classifier = classifier

    def build_classifier(self, pos_features, neg_features, mode):
        knn_train_feats = []
        knn_train_labels = []
        for neg_ele in neg_features:
            knn_train_feats.append(neg_ele)
            knn_train_labels.append(0)

        for pos_ele in pos_features:
            knn_train_feats.append(pos_ele)
            knn_train_labels.append(1)

        if mode == "knn":
            clf = KNeighborsClassifier(
                n_neighbors=4, weights='distance', algorithm='auto', leaf_size=50)
        elif mode == "svm":
            clf = svm.NuSVC(probability=True)
        elif mode == "logistic_reg":
            clf = LogisticRegression(random_state=0)
        else:
            print("Invalid classifier")
        self.classifier = clf.fit(knn_train_feats, knn_train_labels)

    def forward(self, x):
        with torch.no_grad():
            # Get embedding of retrieval image
            x_r = self.model(x)
            pr_o = self.classifier.predict_proba(x_r.cpu().numpy())[:, 1]
            stack = torch.mul(self.masks, x)

            x = []
            for i in range(0, self.N, self.gpu_batch):
                x.append(self.model(stack[i:min(i + self.gpu_batch, self.N)]))
            x = torch.cat(x)
            pr_x = self.classifier.predict_proba(x.cpu().numpy())[:, 1]
            diff = torch.Tensor(pr_o-pr_x).to(x.device)

            # Compute saliency
            K = (1 - self.masks).permute(2, 3, 1, 0) * diff.clamp(min=0)
            sal = self.weighted_avg(K)

        return sal


class FSalBatch(FSal):
    def forward(self, x):
        B, C, H, W = x.size()

        with torch.no_grad():
            # Get embedding of retrieval image
            x_r = self.model(x)
            pr_o = self.classifier.predict_proba(x_r.cpu().numpy())[:, 1]

            # Apply array of masks to the image
            stack = torch.mul(self.masks.view(self.N, 1, H, W),
                              x.view(B * C, H, W))
            stack = stack.view(B * self.N, C, H, W)

            x = []
            for i in range(0, self.N*B, self.gpu_batch):
                x.append(self.model(
                    stack[i:min(i + self.gpu_batch, self.N*B)]))
            x = torch.cat(x)
            pr_x = self.classifier.predict_proba(x.cpu().numpy())[:, 1]
            pr_x = pr_x.reshape(-1, self.N, B).transpose(0,
                                                         2, 1).reshape(-1, self.N)
            diff = torch.Tensor(pr_o[:, None]-pr_x).to(x.device)

            # Compute saliency
            K = (1 - self.masks).permute(2, 3, 1, 0) * diff.clamp(min=0)
            sal = self.weighted_avg(K)

        return sal


class FSalGrad(nn.Module):
    # Adapted from : https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py
    def __init__(self, model, feature_module, target_layers):
        super(FSalGrad, self).__init__()
        self.model = model
        # TODO: Should this be done at initialization stage?
        self.model.fc = nn.Linear(model.fc.in_features, 1)
        self.feature_module = feature_module

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layers, return_gradients=False)

    def get_classifier(self, classifier):
        self.classifier = classifier

        # Load weights into model
        self.model.fc.weight.data = torch.Tensor(self.classifier.coef_)
        self.model.fc.bias.data = torch.Tensor(self.classifier.intercept_)

    def build_classifier(self, pos_features, neg_features, mode):
        knn_train_feats = []
        knn_train_labels = []
        for neg_ele in neg_features:
            knn_train_feats.append(neg_ele)
            knn_train_labels.append(0)

        for pos_ele in pos_features:
            knn_train_feats.append(pos_ele)
            knn_train_labels.append(1)

        if mode == "knn":
            clf = KNeighborsClassifier(
                n_neighbors=4, weights='distance', algorithm='auto', leaf_size=50)
        elif mode == "svm":
            clf = svm.NuSVC(probability=True)
        elif mode == "logistic_reg":
            clf = LogisticRegression(random_state=0)
        else:
            print("Invalid classifier")
        self.classifier = clf.fit(knn_train_feats, knn_train_labels)

        # Load weights into model
        self.model.fc.weight.data = torch.Tensor(self.classifier.coef_)
        self.model.fc.bias.data = torch.Tensor(self.classifier.intercept_)

    def forward(self, x):
        _, _, H, W = x.size()

        # Extract intermediate activations and outputs
        A, x = self.extractor(x)

        # Compute gradients
        feats = A[-1]  # choose last set of features
        grads = torch.autograd.grad(torch.unbind(x), feats)[0]

        with torch.no_grad():
            weights = torch.mean(grads, dim=(2, 3))
            sal = torch.bmm(weights.unsqueeze(1), feats.reshape(
                feats.shape[0], feats.shape[1], -1))
            sal = sal.reshape(feats.shape[0], 1,
                              feats.shape[2], feats.shape[3])

            # Apply ReLU
            sal = sal.clamp(min=0)

            # Upsample
            sal = nn.functional.interpolate(
                sal, size=(H, W), mode='bilinear').squeeze(1)

        return sal


class SimScoreCAM(nn.Module):
    """
    Adapted from: https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py
    """

    def __init__(self, model, feature_module, target_layers, gpu_batch=100):
        super(SimScoreCAM, self).__init__()
        self.model = model
        self.feature_module = feature_module
        self.gpu_batch = gpu_batch

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layers, return_gradients=False)

    def forward(self, x_q, x):
        _, _, H, W = x_q.size()

        with torch.no_grad():
            # Get embedding of query image
            x_q = self.model(x_q)

            # Extract intermediate activations and embedding of retrieval image
            A, x_r = self.extractor(x)
            o_dist = torch.cdist(x_q, x_r)

            # Upsample activations
            activations = A[-1]
            activations = nn.functional.interpolate(
                activations, size=(H, W), mode='bilinear')

            # Compute min/max values
            activations_min = activations.view(
                activations.shape[0], activations.shape[1], -1).min(dim=2).values
            activations_max = activations.view(
                activations.shape[0], activations.shape[1], -1).max(dim=2).values

            # Remove NaN values
            idx = torch.where(activations_min != activations_max)[1]
            activations = activations[:, idx]
            activations_min = activations_min[:, idx, None, None]
            activations_max = activations_max[:, idx, None, None]

            # Normalized saliency map (inverted)
            norm_saliency_map = 1 - \
                (activations - activations_min) / \
                (activations_max - activations_min)
            masked_inputs = torch.matmul(norm_saliency_map.transpose(1, 0), x)

            x = []
            for i in range(0, masked_inputs.shape[0], self.gpu_batch):
                x.append(self.model(
                    masked_inputs[i:min(i + self.gpu_batch, masked_inputs.shape[0])]))
            x = torch.cat(x)
            m_dist = torch.cdist(x_q, x)
            m_dist = (m_dist - o_dist).clamp(min=0)
            # Normalize?
            m_dist = (m_dist - m_dist.min()) / (m_dist.max() - m_dist.min())

            # Mean across dimensions
            score_saliency_map = (
                m_dist[..., None, None] * activations).mean(dim=1)

        return score_saliency_map


class SimAtt(nn.Module):
    def __init__(self, model, feature_module, target_layers):
        super(SimAtt, self).__init__()
        self.model = model
        self.feature_module = feature_module

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layers, return_gradients=False)

    def forward(self, x_q, x_p=None, x_n=None):
        # Consider all possible conditions:
        # 1. anchor + positive (Siamese)
        # 2. anchor + negative (Siamese)
        # 3. anchor + positive + negative (triplet)
        # 4. anchor + positive + negative1 + negative2 (quadruplet)
        _, _, H, W = x_q.size()

        # Concatenate all inputs
        x = x_q
        if x_p is not None:
            x = torch.cat((x, x_p))
        if x_n is not None:
            x = torch.cat((x, x_n))

        # Extract intermediate activations and outputs
        A, x = self.extractor(x)

        # Compute positive and negative weights
        x_norm = nn.functional.normalize(x.detach(), dim=1)
        w = torch.abs(x_norm[0] - x_norm[1:])
        if x_p is not None:
            w[0] = 1 - w[0]

        # Take elementwise product
        w = torch.prod(w, dim=0)

        # Compute sample scores
        s = torch.matmul(torch.abs(x), w)

        # Loop through sample scores
        feats = A[-1]  # choose last set of features
        grads = torch.autograd.grad(torch.unbind(s), feats)[0]

        with torch.no_grad():
            weights = torch.mean(grads, dim=(2, 3))
            M = torch.bmm(weights.unsqueeze(1), feats.reshape(
                feats.shape[0], feats.shape[1], -1))
            M = M.reshape(feats.shape[0], 1, feats.shape[2], feats.shape[3])

            # Apply ReLU
            M = M.clamp(min=0)

            # Upsample
            M = nn.functional.interpolate(
                M, size=(H, W), mode='bilinear').squeeze(1)

        return M


class SimCAM(nn.Module):
    """
    Adapted from: https://github.com/Jeff-Zilence/Explain_Metric_Learning/blob/master/Face_Verification/demo.py
    """

    def __init__(self, model, feature_module, target_layers, fc=None):
        super(SimCAM, self).__init__()
        self.model = model
        self.feature_module = feature_module
        self.fc = fc

        self.extractor = ModelOutputs(
            self.model, self.feature_module, target_layers, return_gradients=False)

    def Point_Specific(self, decom, point=[0, 0], size=(224, 224)):
        """
            Generate the point-specific activation map
            We assume the query point is always on the query image (image 1)
        """
        decom_padding = nn.functional.pad(decom.permute(
            2, 3, 0, 1), (1, 1, 1, 1), mode='replicate').permute(2, 3, 0, 1)

        # Compute the transformed coordinates
        x = (point[0] + 0.5) / size[0] * (decom_padding.shape[0]-2)
        y = (point[1] + 0.5) / size[1] * (decom_padding.shape[1]-2)
        x = x + 0.5
        y = y + 0.5
        x_min = int(np.floor(x))
        y_min = int(np.floor(y))
        x_max = x_min + 1
        y_max = y_min + 1
        dx = x - x_min
        dy = y - y_min
        interpolation = decom_padding[x_min, y_min]*(1-dx)*(1-dy) + \
            decom_padding[x_max, y_min]*dx*(1-dy) + \
            decom_padding[x_min, y_max]*(1-dx)*dy + \
            decom_padding[x_max, y_max]*dx*dy

        return interpolation.clamp(min=0)

    def forward(self, x_q, x, point=None):
        _, _, H, W = x_q.size()

        with torch.no_grad():
            # Concatenate all inputs
            x = torch.cat((x_q, x))

            # Extract intermediate activations and outputs
            A, _ = self.extractor(x)

            # Reshape dimensions
            x = A[-1].permute(0, 2, 3, 1)

            if self.fc is not None:
                x = torch.matmul(x, self.fc.weight.data.transpose(
                    1, 0)) + self.fc.bias.data / (x.shape[1] * x.shape[2])

            Decomposition = torch.zeros(
                [x.shape[1], x.shape[2], x.shape[1], x.shape[2]], device=x_q.device)
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    for k in range(x.shape[1]):
                        for l in range(x.shape[2]):
                            Decomposition[i, j, k, l] = torch.sum(
                                x[0, i, j]*x[1, k, l])
            Decomposition = Decomposition / torch.max(Decomposition)

            # Apply ReLU
            Decomposition = Decomposition.clamp(min=0)

            # Map for query image
            decom_1 = torch.sum(Decomposition, dim=(2, 3))

            # Map for retrieval image
            # Do point specific calculation here if needed
            if point is not None:
                decom_2 = self.Point_Specific(
                    Decomposition, point, size=(H, W))
            else:
                decom_2 = torch.sum(Decomposition, dim=(0, 1))

            # Upsample
            Decomposition = nn.functional.interpolate(torch.stack(
                (decom_1, decom_2)).unsqueeze(1), size=(H, W), mode='bilinear').squeeze(1)

        return Decomposition
