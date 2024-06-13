import torch
import numpy as np

from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.transforms import ToGrid

ALL_FRONT = [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]

ALL_LEFT = [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0]]

FRONT_LEFT = [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]]

FRONT_RIGHT = [[0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]]

MASK_MODES = {
    'ALL_FRONT': ALL_FRONT,
    'ALL_LEFT': ALL_LEFT,
    'FRONT_LEFT': FRONT_LEFT,
    'FRONT_RIGHT': FRONT_RIGHT,
}


def weight_region(eeg, mode='ALL_FRONT', weight_roi=1.0, weight_nroi=-1.0):
    assert mode in MASK_MODES
    mask_list = MASK_MODES[mode]
    if isinstance(eeg, np.ndarray):
        mask = np.array(mask_list, dtype=eeg.dtype)
        nmask = np.abs(1 - mask)
    elif isinstance(eeg, torch.Tensor):
        mask = torch.tensor(mask_list, dtype=eeg.dtype, device=eeg.device)
        nmask = torch.abs(1 - mask)
    else:
        raise NotImplementedError()
    return eeg * (mask * weight_roi + nmask * weight_nroi)


# quick test
# eeg = ToGrid(DEAP_CHANNEL_LOCATION_DICT)(eeg=np.ones((32, 128)))['eeg']
# mask_region(eeg)


def grad_cam_loss_v1(model,
                     inputs,
                     labels,
                     mode='FRONT_LEFT',
                     weight_roi=1.0,
                     weight_nroi=-1.0):
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

    grad_input_tem = torch.autograd.grad(
        outputs=intermediate_outputs[len(intermediate_outputs) - 1],
        grad_outputs=alpher_vector_list[len(alpher_vector_list) - 1],
        inputs=inputs,
        retain_graph=True,
        create_graph=True)[0]

    des_grad_intermediate = torch.sum(
        grad_intermediate_list[len(grad_intermediate_list) - 1],
        dim=(1, 2, 3)).view(-1, 1, 1, 1)
    grad_input = des_grad_intermediate * grad_input_tem

    tem_data = torch.sum(grad_input, axis=1)

    max_matrix = torch.amax(tem_data, dim=(1, 2))
    min_matrix = torch.amin(tem_data, dim=(1, 2))
    min_matrix = min_matrix.view(min_matrix.shape[0], 1, 1)
    max_matrix = max_matrix.view(max_matrix.shape[0], 1, 1)
    tem_data = (tem_data - min_matrix) / (max_matrix - min_matrix)
    tem_data = torch.sum(tem_data, axis=0)

    mul_matrix = weight_region(tem_data,
                               mode=mode,
                               weight_roi=weight_roi,
                               weight_nroi=weight_nroi)
    return torch.sum(mul_matrix)

if __name__ == '__main__':
    # quick test
    from model import Model

    mock_model = Model(num_classes=2)
    mock_input = torch.randn(2, 128, 9, 9)
    mock_y = torch.ones(2, dtype=torch.long)

    print(grad_cam_loss_v1(mock_model, mock_input, mock_y))