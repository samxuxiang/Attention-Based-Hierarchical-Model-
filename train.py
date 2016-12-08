from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.vgg19 import Vgg19


def main():
    # load train dataset
    data = load_coco_data(data_path='./data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data', split='val')

    model = CaptionGenerator(word_to_idx, dim_layer1=[7,512],dim_layer2=[14,512], dim_embed=512,
                             dim_hidden=1024, n_time_step=16,  alpha_c=[1.0,1.0] ,alpha_e=[5.0,10.0])

    solver = CaptioningSolver(model, data, val_data,
                              n_epochs=20, batch_size=32, update_rule='adam', learning_rate=0.001, 
                              print_every=100, save_every=1, image_path='./image/',
                              pretrained_model=None, model_path='model6/', test_model=None,
                              print_bleu=True, log_path='log6/',gpu='/gpu:3')
    
    print "start training.."
    solver.train()

if __name__ == "__main__":
    main()

