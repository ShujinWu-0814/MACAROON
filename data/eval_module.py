from torch.utils.data import DataLoader, Dataset
import torch
import jsonlines

    
class EvalModule():
    def collate_fn(self, batch):
        if len(batch) != 1:
            raise ValueError("Batch size must be 1")
        img_id = batch[0]["Image ID"]
        img_path = '../datasets/eval/images/' + img_id + '.jpg'
        question = batch[0]['Question']
        return {
            'image_path': img_path,
            'question': question,
        }
        
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_data,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
    
    def read_jsonl(self, path):
        with jsonlines.open(path, 'r') as reader:
            save_results = [obj for obj in reader]
        return save_results

    def __init__(self, config: dict):
        self.config = config
        self.eval_data_path = config["data"]["eval_data_path"]
        self.eval_data = self.read_jsonl(self.eval_data_path) # [{}, {}, {}, {}]

