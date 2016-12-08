import matplotlib.pyplot as plt
import cPickle as pickle
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.bleu import evaluate

plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'




data = load_coco_data(data_path='./data', split='val')
with open('./data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)


model = CaptionGenerator(word_to_idx, dim_layer1=[7,512],dim_layer2=[14,512], dim_embed=512,
                             dim_hidden=1024, n_time_step=16,  alpha_c=[1.0,1.0] ,alpha_e=[5.0,5.0])

solver = CaptioningSolver(model, data, data,
                              n_epochs=20, batch_size=32, update_rule='adam', learning_rate=0.001, 
                              print_every=100, save_every=1, image_path='./image/val2014_resized',
                              pretrained_model=None, model_path='model6/', test_model='model6/model-6',
                              print_bleu=True, log_path='log5/',gpu='/gpu:0')
print "start testing.."

solver.visualize_samples(data, 'val',10)
