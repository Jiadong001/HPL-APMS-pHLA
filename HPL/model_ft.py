'''
fine-tune model
'''
import torch
import torch.nn as nn

from tape import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP

class ProjectionHead(nn.Module):
    def __init__(self, input_size: int, num_labels: int, head_type: str):
        super().__init__()
        self.head_type = head_type
        
        if head_type == "2mlp":
            self.classifier = SimpleMLP(input_size, 512, num_labels)     

    def forward(self, pooled_output):
        outputs = self.classifier(pooled_output)
        return outputs


class meanTAPE(ProteinBertAbstractModel):
    def __init__(self, tape_config, head_type):
        super().__init__(tape_config)
        self.tape = ProteinBertModel.from_pretrained('bert-base')
        self.projection = ProjectionHead(tape_config.hidden_size, 2, head_type)

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.tape(input_ids, input_mask=input_mask)[0]        # sequence_output
        outputs = torch.mean(outputs, dim=1)                            # mean of sequence_output
        outputs = self.projection(outputs)

        return outputs


class clsTAPE(ProteinBertAbstractModel):
    def __init__(self, tape_config, head_type):
        super().__init__(tape_config)
        self.tape = ProteinBertModel.from_pretrained('bert-base')
        self.projection = ProjectionHead(tape_config.hidden_size, 2, head_type)

    def forward(self, input_ids, input_mask=None, targets=None):

        cls_outputs = self.tape(input_ids, input_mask=input_mask)[0][:,0]  # output of every sample's <cls> token
        outputs = self.projection(cls_outputs)                     

        return outputs
