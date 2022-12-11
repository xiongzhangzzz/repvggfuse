# test phase
import torch
from torch.autograd import Variable
from net import RepVGG_Fuse_net
import utils
from args_fusion import get_parser
import numpy as np
import os


def load_model(path, input_nc, output_nc):

	nest_model = RepVGG_Fuse_net(input_nc, output_nc)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
	# encoder
	en_r = model.encoder(img1)
	en_v = model.encoder(img2)
	# fusion
	f = model.fusion(en_r, en_v, strategy_type=strategy_type)

	# decoder
	img_fusion = model.decoder(f)
	return img_fusion[0]


def run_demo(model, infrared_path, visible_path, output_path_root, strategy_type, mode, args, name, ssim_weight_str):

	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode)


	# dim = img_ir.shape
	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)
	dimension = ir_img.size()

	img_fusion = _generate_fusion_image(model, strategy_type, ir_img, vis_img)
	############################ multi outputs ##############################################
	file_name =  str(name[:-4]) + '_repvgg_cn_' + strategy_type + '_' + str(ssim_weight_str) + str(name[-4:])
	output_path = os.path.join(output_path_root, file_name)
	# # save images
	# utils.save_image_test(img_fusion, output_path)
	# utils.tensor_save_rgbimage(img_fusion, output_path)
	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	utils.save_images(output_path, img)

	print(output_path)


def main():
	parsers = get_parser()
	args = parsers.parse_args()
	# run demo
	test_path = args.test_path


	in_c = 1
	out_c = in_c
	mode = 'L'
	model_path = args.model_path


	ssim_weight_str = model_path.split('/')[-1][:-6].split('_')[-1]
	strategy_type_list = ['addition', 'attention_weight']  # addition, attention_weight
	output_path = os.path.join(args.output_path, str(ssim_weight_str))
	strategy_type = strategy_type_list[0]

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)
	with torch.no_grad():
		print('SSIM weight ----- ' + ssim_weight_str)
		model = load_model(model_path, in_c, out_c)
		test_file = os.listdir(os.path.join(test_path, 'ir'))
		for name in test_file:
			infrared_path = os.path.join(test_path, 'ir/' + name)
			visible_path = os.path.join(test_path,  'vi/' +name)
			run_demo(model, infrared_path, visible_path, output_path, strategy_type, mode, args, name, ssim_weight_str)
	print('Done......')

if __name__ == '__main__':
	main()
