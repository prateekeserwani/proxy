import os
import glob
import cv2
import moviepy.video.io.ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt

image_folder='./temp/'
fps=1
arch=0
no_of_epoch =49

collection=[]

count=0
#while(1):
#print(len(os.listdir(os.path.join(image_folder,str(arch)+'_0', str(count)+'_*.jpg'))))
jpgFilenamesList = glob.glob(os.path.join(image_folder,str(arch)+'_1', str(count)+'_*.jpg'))
print(len(jpgFilenamesList))

kernel_group=0
while(1):
	jpgFilenamesList = glob.glob(os.path.join(image_folder,str(arch)+'_1', str(kernel_group)+'_*.jpg'))
	if len(jpgFilenamesList)==0:
		break
	epoch_list=[1,15,30,49]
	for i in range(len(jpgFilenamesList)):
		image=[]
		for j, epoch in enumerate(epoch_list):
			filename=os.path.join(image_folder,str(arch)+'_'+str(epoch), str(kernel_group)+'_'+str(i)+'.jpg')
			readimage = cv2.imread(filename)
			height, width, channel = readimage.shape
			highlight_string = 'Epoch =>'+str(epoch) 
			cv2.putText(readimage,highlight_string, (width//3,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
			#plt.imshow(readimage)
			#plt.show()
			image.append(readimage)
		image = np.concatenate(image,axis=1)
		print(image.shape)
		collection.append(image)
		print(filename)
	kernel_group=kernel_group+1
	

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(collection, fps=fps)
clip.write_videofile('./video/architecture_'+str(arch)+'.mp4')

'''
video = cv2.VideoWriter('my_video.mp4', 0, 1, (100,200))

for image in collection:
	print(image.shape)
	video.write(image)

cv2.destroyAllWindows()
video.release()

		
'''
