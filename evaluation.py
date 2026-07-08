import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


n_classes = 1000


def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn, input_size=224):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            input_size (int): size of input image (default: 224).
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.hw = input_size * input_size  # image area

    def single_run(self, img_tensor, retrieved_tensor, explanation, verbose=0, save_to=None, batch_size=64):
        r"""Run metric on one image-saliency pair.

        Pixels are revealed/removed ``self.step`` at a time in order of
        decreasing saliency. Instead of running the model once per step
        (batch size 1), all masked variants are materialised and pushed
        through the model in mini-batches, which is numerically identical
        but far faster on the GPU.

        Args:
            img_tensor (Tensor): normalized query image tensor.
            retrieved_tensor (Tensor): normalized retrieved image tensor.
            explanation (np.ndarray | Tensor): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
            batch_size (int): number of masked steps evaluated per forward pass.

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        with torch.no_grad():
            device = retrieved_tensor.device
            q_feat = self.model(img_tensor.to(device))
            n_steps = (self.hw + self.step - 1) // self.step

            if self.mode == 'del':
                title = 'Deletion game'
                ylabel = 'Pixels deleted'
                start = retrieved_tensor.clone()
                finish = self.substrate_fn(retrieved_tensor)
            elif self.mode == 'ins':
                title = 'Insertion game'
                ylabel = 'Pixels inserted'
                start = self.substrate_fn(retrieved_tensor)
                finish = retrieved_tensor.clone()

            side = int(round(self.hw ** 0.5))
            start_flat = start.reshape(3, self.hw).to(device)
            finish_flat = finish.reshape(3, self.hw).to(device)

            # Pixel indices ordered from most to least salient.
            if isinstance(explanation, torch.Tensor):
                t_r = explanation.detach().cpu().numpy()
            else:
                t_r = np.asarray(explanation)
            t_r = t_r.reshape(-1, self.hw)
            salient_order = np.argsort(t_r, axis=1)          # ascending saliency
            order_desc = np.ascontiguousarray(salient_order[:, ::-1]).reshape(-1)
            order_flat = torch.from_numpy(order_desc).long().to(device)

            # reveal_img_idx[p] = first step index at which pixel p equals `finish`.
            # Pixel at position `pos` in the ordering is revealed after step pos//step.
            positions = torch.arange(self.hw, device=device)
            reveal_img_idx = torch.empty(self.hw, dtype=torch.long, device=device)
            reveal_img_idx[order_flat] = positions // self.step + 1

            scores = np.empty(n_steps + 1)
            zero_cntr = 0
            step_indices = torch.arange(n_steps + 1, device=device)

            for b in range(0, n_steps + 1, batch_size):
                idx = step_indices[b:b + batch_size]                 # (bs,)
                # mask[j, p] = True -> pixel p is `finish` for step idx[j].
                mask = reveal_img_idx.unsqueeze(0) <= idx.unsqueeze(1)  # (bs, hw)
                imgs = torch.where(
                    mask.unsqueeze(1),
                    finish_flat.unsqueeze(0),
                    start_flat.unsqueeze(0),
                ).reshape(-1, 3, side, side)                          # (bs, 3, H, W)

                feats = self.model(imgs)
                sims = torch.nn.functional.cosine_similarity(q_feat, feats)  # (bs,)
                zero_cntr += int((sims < 0).sum().item())
                sims = torch.clamp(sims, 0.0, 1.0)
                scores[b:b + idx.shape[0]] = sims.detach().cpu().numpy()

            if verbose or save_to:
                for i in range(n_steps + 1):
                    if not (verbose == 2 or (verbose == 1 and i == n_steps) or save_to):
                        continue
                    cur = torch.where(
                        (reveal_img_idx <= i).unsqueeze(0),
                        finish_flat, start_flat,
                    ).reshape(3, side, side)
                    plt.figure(figsize=(10, 5))
                    plt.subplot(121)
                    plt.title('{} {:.1f}%, P={:.4f}'.format(
                        ylabel, 100 * i / n_steps, scores[i]))
                    plt.axis('off')
                    tensor_imshow(cur.cpu())

                    plt.subplot(122)
                    plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1])
                    plt.xlim(-0.1, 1.1)
                    plt.ylim(0, 1.05)
                    plt.fill_between(np.arange(i + 1) / n_steps,
                                     0, scores[:i + 1], alpha=0.4)
                    plt.title(title)
                    plt.xlabel(ylabel)
                    if save_to:
                        plt.savefig(save_to + '/{:03d}.png'.format(i))
                        plt.close()

        return auc(scores), zero_cntr

    def evaluate_similarity(self, img_batch,  retrieved_batch, exp_batch, batch_size, k=10):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        n_classes = img_batch.shape[1]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            q_feats = self.model(
                img_batch[i*batch_size:(i+1)*batch_size]).cpu()
            predictions[i*batch_size:((i+1)*batch_size)] = q_feats
        n_steps = (self.hw + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        t_r = exp_batch.reshape(-1, self.hw)
        salient_order = np.argsort(t_r, axis=1)
        salient_order = torch.flip(salient_order, [0, 1])

        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        n_samples = retrieved_batch.shape[0]
        n_classes = retrieved_batch.shape[1]
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(
                retrieved_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = retrieved_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = retrieved_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                new_ret_feat = self.model(start[j*batch_size:(j+1)*batch_size])
                c_dist = torch.nn.functional.cosine_similarity(
                    predictions[j*batch_size:(j+1)*batch_size], new_ret_feat)
                c_dist = torch.clamp(c_dist, min=0, max=1)
                scores[i, j*batch_size:(j+1)*batch_size] = c_dist
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, self.hw)[
                r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, self.hw)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores, auc(scores.mean(1))

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i*batch_size:(i+1)*batch_size]).cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (self.hw + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        t_r = exp_batch.reshape(-1, self.hw)
        salient_order = np.argsort(t_r, axis=1)
        salient_order = torch.flip(salient_order, [0, 1])

        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(
                img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(start[j*batch_size:(j+1)*batch_size])
                preds = preds.cpu().numpy()[range(
                    batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, 3, self.hw)[
                r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, self.hw)[r, :, coords]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores, auc(scores.mean(1))
