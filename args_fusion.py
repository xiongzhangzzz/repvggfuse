import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=4)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--dataset", type=str, default="./mscoco database/train2014")
	parser.add_argument("--HEIGHT", type=int, default=256)
	parser.add_argument("--WIDTH",type=int, default=256)
	parser.add_argument("--save_model_dir", type=str, default="./models")
	parser.add_argument("--save_loss_dir", type=str, default="./models/loss")
	parser.add_argument("--image_size", type=int, default=256)
	parser.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
	parser.add_argument("--seed", type=float, default=42, help="random seed for training")
	parser.add_argument("--ssim_weight", type=int, default=[1,10,100,1000,10000])
	parser.add_argument("--ssim_path", type=int, default=['1e0', '1e1', '1e2', '1e3', '1e4', '1e5'])

	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--lr_light", type=float, default=1e-4)
	parser.add_argument("--log_interval", type=int, default=5, help="number of images after which the training loss is logged, default is 5")


	parser.add_argument("--resume", default=None)
	parser.add_argument("--resume_auto_en", default=None)
	parser.add_argument("--resume_auto_de", default=None)
	parser.add_argument("--resume_auto_fn", default=None)
	
	parser.add_argument("--model_path", type=str, default="/media/alex/D/learn/code/mymethod/repvggfuse_1.0/repvggfuse_v1.7_final/models/1e0/Final_epoch_1e0.model")
	
	parser.add_argument("--test_path", type = str, default='/media/alex/D/learn/code/mymethod/repvggfuse_1.0/repvggfuse_v1.7_final/images/')
	parser.add_argument("--output_path", type = str, default='/media/alex/D/learn/code/mymethod/repvggfuse_1.0/repvggfuse_v1.7_final/output')
	return parser




