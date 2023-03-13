from transformers import BertModel, DPRConfig, DPRContextEncoder, DPRQuestionEncoder
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        pooler_output = self.encoder(input_ids, attention_mask).pooler_output
        return pooler_output

    def save_pretrained(self, address):
        self.encoder.save_pretrained(address)


class BiEncoder(torch.nn.Module):
    def __init__(self, args):
        ctx_model = args.ctx_model_path
        qry_model = args.qry_model_path
        super().__init__()
        self.ctx_model = BertModel.from_pretrained(ctx_model)
        self.qry_model = BertModel.from_pretrained(qry_model)
        config = DPRConfig()
        config.vocab_size = self.ctx_model.config.vocab_size
        context_encoder = DPRContextEncoder(config)
        context_encoder.base_model.base_model.load_state_dict(self.ctx_model.state_dict(), strict=False)
        query_encoder = DPRQuestionEncoder(config)
        query_encoder.base_model.base_model.load_state_dict(self.qry_model.state_dict(), strict=False)
        self.ctx_model = context_encoder
        self.qry_model = query_encoder
        self.qry_model = EncoderWrapper(self.qry_model)
        self.ctx_model = EncoderWrapper(self.ctx_model)
        self.encoder_gpu_train_limit = args.encoder_gpu_train_limit

    def encode(self, model, input_dict):
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        all_pooled_output = []
        for sub_bndx in range(0, input_ids.shape[0], self.encoder_gpu_train_limit):
            sub_input_ids = input_ids[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.encoder_gpu_train_limit]
            pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
            all_pooled_output.append(pooler_output)
        return torch.cat(all_pooled_output, dim=0)

    def forward(self, data):
        ctx_vector = self.encode(self.ctx_model, data['ctx'])
        qry_vector = self.encode(self.qry_model, data['qry'])
        dot_products = torch.matmul(qry_vector, ctx_vector.transpose(0, 1))
        probs = F.log_softmax(dot_products, dim=1)
        loss = F.nll_loss(probs, data['positive'].long())
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == data['positive']).sum() / data['positive'].shape[0]
        return loss, accuracy

    def save_pretrained(self, addr):
        self.ctx_model.save_pretrained(addr + '/ctx')
        self.qry_model.save_pretrained(addr + '/qry')
