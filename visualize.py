import torch
import os

from PIL import Image
from torcheeg.utils import plot_2d_tensor
from torcheeg.utils import plot_3d_tensor
from model import Model


def visualize_grad_cam(model, inputs, labels):
    intermediate_outputs = {}

    def hook(module, input, output):
        intermediate_outputs[module] = output

    handle = model.selu1.register_forward_hook(hook)

    outputs = model(inputs.requires_grad_())
    intermediate_outputs = list(intermediate_outputs.values())

    handle.remove()
    # start here
    grad_output = torch.nn.functional.one_hot(labels)

    alpher_vector_list = []
    grad_intermediate_list = []

    for i in range(len(intermediate_outputs)):
        if i == 0:
            grad_intermediate = torch.autograd.grad(
                outputs=outputs,
                inputs=intermediate_outputs[i],
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=True)[0]
        else:
            grad_intermediate = torch.autograd.grad(
                outputs=intermediate_outputs[i - 1],
                inputs=intermediate_outputs[i],
                grad_outputs=alpher_vector_list[i - 1],
                retain_graph=True,
                create_graph=True)[0]
            grad_intermediate = grad_intermediate * torch.sum(
                grad_intermediate_list[i - 1], dim=(1, 2, 3)).view(-1, 1, 1, 1)

        alpher_vector = torch.sum(grad_intermediate,
                                  axis=[2, 3]).view(grad_intermediate.shape[0],
                                                    grad_intermediate.shape[1],
                                                    1, 1)

        max_alpher = torch.amax(alpher_vector, dim=(1, 2, 3))
        min_alpher = torch.amin(alpher_vector, dim=(1, 2, 3))
        min_alpher = min_alpher.view(min_alpher.shape[0], 1, 1, 1)
        max_alpher = max_alpher.view(max_alpher.shape[0], 1, 1, 1)

        alpher_vector = ((alpher_vector - min_alpher) /
                         (max_alpher - min_alpher))

        alpher_vector = torch.tile(
            alpher_vector,
            (1, 1, grad_intermediate.shape[2], grad_intermediate.shape[3]))
        alpher_vector[torch.clamp(alpher_vector, min=0.9, max=1) == 0.9] = 0
        alpher_vector_list.append(alpher_vector)
        grad_intermediate_list.append(grad_intermediate)

    guided_backpropgation = torch.autograd.grad(
        outputs=intermediate_outputs[len(intermediate_outputs) - 1],
        grad_outputs=alpher_vector_list[len(alpher_vector_list) - 1],
        inputs=inputs,
        retain_graph=True,
        create_graph=True)[0]

    grad_cam = torch.sum(grad_intermediate_list[len(grad_intermediate_list) -
                                                1],
                         dim=1,
                         keepdim=True)

    guided_grad_cam = grad_cam * guided_backpropgation
    # guided_backpropgation (guided backpropgation) [2, 128, 9, 9]
    # grad_cam (grad cam) [2, 1, 9, 9]
    # guided_grad_cam (guided grad cam) [2, 128, 9, 9]
    return guided_backpropgation, grad_cam, guided_grad_cam


if __name__ == '__main__':
    # # quick test
    # from model import Model

    # mock_model = Model(num_classes=2)
    # mock_input = torch.randn(2, 128, 9, 9)
    # mock_y = torch.ones(2, dtype=torch.long)

    # print(visualize_grad_cam(mock_model, mock_input, mock_y))

    from torcheeg import transforms
    from torcheeg.datasets import DEAPDataset
    from torcheeg.datasets.constants.emotion_recognition.deap import \
        DEAP_CHANNEL_LOCATION_DICT

    PARAMS = {
        'IO_PATH': './tmp_out/xai/deap',
        'ROOT_PATH': './tmp_in/data_preprocessed_python',
        'LOG_PATH': './tmp_out/xai_log/',
        'PARAM_PATH': './tmp_out/xai_log/',
        'SPLIT_PATH': './tmp_out/xai/split',
        'NUM_CLASSES': 2,
        'BATCH_SIZE': 256,
        'EPOCH': 50,
        'KFOLD': 10,
        'DEVICE_IDS': [1],
        'WEIGHT_DECAY': 1e-4,
        'LR': 1e-4,
        'LABEL': 'valence'
    }

    dataset = DEAPDataset(
        io_path=PARAMS['IO_PATH'],
        root_path=PARAMS['ROOT_PATH'],
        offline_transform=transforms.Compose([
            transforms.BaselineRemoval(),
            transforms.MeanStdNormalize(),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ]),
        online_transform=transforms.Compose([transforms.ToTensor()]),
        label_transform=transforms.Compose(
            [transforms.Select(PARAMS['LABEL']),
             transforms.Binary(5.0)]),
        num_worker=16)

    X, y = dataset[0]

    model = Model(num_classes=PARAMS['NUM_CLASSES'])
    model.load_state_dict(torch.load(os.path.join(PARAMS['PARAM_PATH'], f'2022-09-26-12-01-07.pt'))['model'])

    guided_backpropgation, grad_cam, guided_grad_cam = visualize_grad_cam(
        model, X.unsqueeze(0), torch.tensor([y], dtype=torch.long))

    guided_backpropgation = guided_backpropgation[0]
    grad_cam = grad_cam[0][0]
    guided_grad_cam = guided_grad_cam[0]

    img = plot_3d_tensor(X)
    img = Image.fromarray(img)
    img.save("raw_input.png")

    img_1 = plot_3d_tensor(guided_backpropgation.detach())
    img_1 = Image.fromarray(img_1)
    img_1.save("guided_backpropgation.png")

    img_2 = plot_3d_tensor(guided_grad_cam.detach())
    img_2 = Image.fromarray(img_2)
    img_2.save("guided_grad_cam.png")

    img_3 = plot_2d_tensor(grad_cam.detach())
    img_3 = Image.fromarray(img_3)
    img_3.save("grad_cam.png")

    print(y)