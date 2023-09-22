class Hi:
    def __init__(self, c):
        self.c = c
from multiprocessing import cpu_count
print(cpu_count())

# print("train label mask\n", self.label_bert_masks["train"][0])
# print("train bert mask\n", self.label_bert_masks["train"][1])
#
# print("val label mask\n", self.label_bert_masks["val"][0])
# print("val bert mask\n", self.label_bert_masks["val"][1])
#
# print("test label mask\n", self.label_bert_masks["test"][0])
# print("test bert mask\n", self.label_bert_masks["test"][1])
# exit(0)


       # for mode in ["train", "val", "test"]:
       #      print(f"{mode} input_one_hot_label_arr")
       #      print((dataset.model_input[mode][0][0]))
       #      print(f"{mode} encoded_input_arr")
       #      print(torch.hstack(dataset.model_input[mode][0][1:]))
       #      print(f"{mode} masked_one_hot_label_arr")
       #      print((dataset.model_input[mode][1][0]))
       #      print(f"{mode} masked_input_arr")
       #      print(torch.hstack(dataset.model_input[mode][1][1:]))
       #      print(f"{mode} label_mask")
       #      print((dataset.model_input[mode][2]))
       #      print(f"{mode} augm_mask")
       #      print((dataset.model_input[mode][3]))
       #      print(f"{mode} target loss")
       #      print((dataset.model_input[mode][4]))
       #  exit(0)