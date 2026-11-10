import torch
from torchvision import transforms


class PGD:
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=8 / 255,
        steps=10,
        step_size=2 / 255,
        norm_type="linfty",
        random_init=True,
        *args,
        **kwargs,
    ):
        self.model_name = model_name
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.norm_type = norm_type
        self.random_init = random_init
        self.device = next(self.model.parameters()).device
        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def attack(self, images, texts, *args):
        images = images.clone().detach().to(self.device)
        delta = self.init_delta(images)

        with torch.no_grad():
            orig_images_feat = self.get_image_feat(images).detach()

        for _ in range(self.steps):
            adv_images_feat = self.get_image_feat(self.transform(images + delta))
            loss = self.get_loss(adv_images_feat, orig_images_feat)
            grad = self.get_grad(loss, delta)
            delta = self.update_delta(delta, images, grad)

        return (images + delta).detach(), texts

    def get_image_feat(self, images):
        image_output = self.model.inference_image(self.normalize(images))
        image_feat = image_output["image_feat"]
        return image_feat

    def transform(self, images, **kwargs):
        return images

    def get_loss(self, adv_images_feat, orig_images_feat):
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        loss = criterion(adv_images_feat.log_softmax(dim=-1), orig_images_feat.softmax(dim=-1))
        return loss

    def get_grad(self, loss, delta):
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def init_delta(self, images):
        delta = torch.zeros_like(images).to(self.device)
        if self.random_init:
            if self.norm_type == "linfty":
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=-1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(images).uniform_(0, 1).to(self.device)
                delta *= r / n * self.epsilon
            delta = self.clamp(delta, -images, 1 - images)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, images, grad):
        if self.norm_type == "linfty":
            delta = torch.clamp(delta + self.step_size * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (
                (delta + scaled_grad * self.step_size)
                .view(delta.size(0), -1)
                .renorm(p=2, dim=0, maxnorm=self.epsilon)
                .view_as(delta)
            )
        delta = self.clamp(delta, -images, 1 - images)
        return delta.detach().requires_grad_(True)

    def clamp(self, x, x_min, x_max):
        return torch.min(torch.max(x, x_min), x_max)

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.attack(*input, **kwargs)
