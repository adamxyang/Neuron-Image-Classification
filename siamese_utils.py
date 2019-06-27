import torch
from torch import nn
import torch.nn.functional as F


class PseudoSiamese(nn.Module):
	def __init__(self, *args, **kwargs):
		super(PseudoSiamese, self).__init__()
		self.leg_a = self._make_siamese_leg()
		self.leg_b = self._make_siamese_leg()
		self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.fc1 = nn.Linear(512, 512)
		self.dropout = nn.Dropout(p=0.2)
		self.fc2 = nn.Linear(512, 2)
		self.leg_a.apply(self._init_weights)
		self.leg_b.apply(self._init_weights)
		self.apply(init_weights)
	def _make_siamese_leg(self):
		def _make_vgg_block(layers=[], input_depth=1, depth=32, kernel_size=3, padding=True, max_pool=True):
			pad = kernel_size//2 if padding else 0
			layers.append( nn.Conv2d(input_depth, depth, kernel_size=kernel_size, padding=pad) )
			layers.append( nn.ReLU() )
			layers.append( nn.BatchNorm2d(depth) )
			layers.append( nn.Conv2d(depth, depth, kernel_size=kernel_size, padding=pad) )
			layers.append( nn.ReLU() )
			layers.append( nn.BatchNorm2d(depth) )
			if max_pool:
				layers.append( nn.MaxPool2d(2) )
			return layers
		layers = []
		layers = _make_vgg_block(layers=layers, input_depth=1, depth=32, kernel_size=3, padding=True, max_pool=True)
		layers = _make_vgg_block(layers=layers, input_depth=32, depth=64, kernel_size=3, padding=True, max_pool=True)
		layers = _make_vgg_block(layers=layers, input_depth=64, depth=128, kernel_size=3, padding=True, max_pool=True)
		layers = _make_vgg_block(layers=layers, input_depth=128, depth=128, kernel_size=3, padding=True, max_pool=False)
		layers.append( nn.Dropout2d(p=0.25) )
		return nn.Sequential(*layers)
	def forward(self, x1, x2):
		x1 = self.leg_a(x1)
		x2 = self.leg_b(x2)
		fts = torch.cat([x1, x2], dim=1)
		fts = self.conv1(fts)
		fts = self.bn1( F.relu(fts) )
		fts = self.conv2(fts)
		fts = self.bn2( F.relu(fts) )
		fts = F.max_pool2d(fts, 2)
		fts = fts.view(fts.size(0), -1)
		fts = self.fc1(fts)
		fts = self.dropout(fts)
		fts = F.relu(fts)
		fts = self.fc2(fts)
		if self.training:
			return fts
		else:
			return torch.softmax(fts, dim=1)



class mydataset(Dataset):
	"""
	Arguments:
		Path to image folder
		Extension of images
		PIL transforms
	"""

	def __init__(self, df, img_path, transform=None):
	
		tmp_df = df
		assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x )).all(), \
"Some images referenced in the CSV file were not found"
		
		self.img_path = img_path
		self.transform = transform

		self.X_train = tmp_df['filenames']
		self.y_train = tmp_df['labels']

	def __getitem__(self, index):
		img = Image.open(self.img_path + self.X_train[index])
		img = img.convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		
		label = torch.from_numpy(self.y_train[index])
		return img, label

	def __len__(self):
		return len(self.X_train.index)
		
		