import os
import time
from metrics import *
from skimage import io
from image_proc import *

class Generic_train_test():
	def __init__(self, model, opts, dataloader, logger):
		self.model=model
		self.opts=opts
		self.dataloader=dataloader
		self.logger=logger

	def decode_input(self, data):
		raise NotImplementedError()

	def train(self):
		total_steps = 0
		print('#training images ', len(self.dataloader)*self.opts.batch_sz)

		for epoch in range(self.opts.start_epoch, self.opts.max_epochs):
			if epoch > self.opts.lr_start_epoch_decay - self.opts.lr_step:
				self.model.update_lr()

			if epoch % self.opts.save_freq==0:
				self.model.save_checkpoint(str(epoch))

			for i, data in enumerate(self.dataloader):
				total_steps+=1
				_input=self.decode_input(data)

				self.model.set_input(_input)
				self.model.optimize_parameters()

				#=========== visualize results ============#
				if total_steps % self.opts.log_freq==0:
					info = self.model.get_current_scalars()
					for tag, value in info.items():
						self.logger.add_scalar(tag, value, total_steps)

					results = self.model.get_current_visuals()
					for tag, images in results.items():
						self.logger.add_images(tag, images, total_steps)

					print('epoch', epoch, 'steps', total_steps)
					print('losses', info)

	def test_individual(self, _input, dir_results, i):
		self.model.set_input(_input)

		start_time = time.time()
		im_pred = self.model.eval()

		# for efficiency measurement
		_ = im_pred.cpu()
		diff_time = time.time() - start_time

		print('processing frame %d' % i)

		# save images 
		if self.opts.save_images:
			np_im_in = white_balance(self.model.im_blur.detach().cpu().numpy().transpose(0,2,3,1)[0].clip(0.,1.))
			np_im_pred = white_balance(im_pred.detach().cpu().numpy().transpose(0,2,3,1)[0].clip(0.,1.))
			
			io.imsave(os.path.join(dir_results, '%04d_blur.png' % i), np_im_in)
			io.imsave(os.path.join(dir_results, '%04d_pred.png' % i), np_im_pred)
			if self.opts.compute_metrics:
				np_im_gt = white_balance(self.model.im_target.detach().cpu().numpy().transpose(0,2,3,1)[0].clip(0.,1.))
				io.imsave(os.path.join(dir_results, '%04d_gt.png' % i), np_im_gt)

		return im_pred, diff_time

	def test(self):
		sum_psnr=0.
		sum_ssim=0.
		sum_time=0.
		f_metric_all=None
		f_metric_avg=None
		n_frames=0
		dir_results=os.path.join(self.opts.results_dir)
		if not os.path.exists(dir_results):
			os.makedirs(dir_results)

		if self.opts.compute_metrics:
			f_metric_all=open(os.path.join(dir_results, 'metric_all'), 'w')
			f_metric_avg=open(os.path.join(dir_results, 'metric_avg'), 'w')

			f_metric_all.write('# frame_id, PSNR_pred, SSIM_pred, time (seconds)\n')
			f_metric_avg.write('# avg_PSNR_pred, avg_SSIM_pred, time (seconds)\n')

		for i, data in enumerate(self.dataloader):
			_input=self.decode_input(data)
			im_pred, diff_time = self.test_individual(_input, dir_results, i)

			# compute metrics 
			if self.opts.compute_metrics:
				psnr_pred=PSNR(im_pred, self.model.im_target)
				ssim_pred=SSIM(im_pred, self.model.im_target)

				sum_psnr += psnr_pred
				sum_ssim += ssim_pred
				sum_time += diff_time
				n_frames += 1

				print('PSNR(%.2f dB) SSIM(%.2f) time(%.2f seconds)\n' % (psnr_pred, ssim_pred, diff_time))
				f_metric_all.write('%d %.2f %.2f %.2f\n' % (i, psnr_pred, ssim_pred, diff_time))

		if self.opts.compute_metrics:
			psnr_avg = sum_psnr / n_frames
			ssim_avg = sum_ssim / n_frames
			time_avg = sum_time / n_frames

			print('PSNR_avg (%.2f dB) SSIM_avg (%.2f) time_avg(%.2f seconds)' % (psnr_avg, ssim_avg, time_avg))
			f_metric_avg.write('%.2f %.2f %.2f\n' % (psnr_avg, ssim_avg, time_avg))

			f_metric_all.close()
			f_metric_avg.close()


