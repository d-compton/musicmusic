# -*- coding: utf-8 -*-

#based on the librosa documentation
#change directory and file inputs for other sets of data

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

#import sklearn.cluster

import librosa
import librosa.display

stop

music_dir = '/home/dcompton/Music/'
BINS_PER_OCTAVE = 12 * 3
N_OCTAVES = 7
#artist = 'Daft Punk/'
#artist_dir = music_dir+artist
years = np.arange(1960,2016,5)
#years = os.listdir(artist_dir)
#y, sr = librosa.load(music_dir +'1960-001 Percy Faith - Theme From A \'Summer Place\'.mp3')

'''
arr_2015 = []


g = open(music_dir+'2015.list','r')
for line in g:
	line = line.split()
	arr_2015.append(int(line[1]))
g.close
'''
print years
f=open(music_dir+'daft_punk2.data','w')

for year in years:
	year_dir = artist_dir+str(year)+'/'
	song_list = os.listdir(year_dir)


	for song in song_list:
		num = int(song[0:2]) #The Beatles

		if year in [1990, 1995]: 
			num = song[3:5]#int()
			if num == 'A1': num = 100
			num = int(num)

		if year in [2010]: num = int(song[0:3])

		if year in [2015]:
			song_t = song
			num = int(song_t.split('.')[0])

		if year in [1960, 1965, 1970, 1975, 1980, 1985, 2000, 2005]:
			num = int(song[5:8])

		'''
		if num in arr_2015:
			print num,song,'already processed'
			continue
		'''

		y, sr = librosa.load(year_dir+song)


		C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
		                                        bins_per_octave=BINS_PER_OCTAVE,
		                                        n_bins=N_OCTAVES * BINS_PER_OCTAVE),
		                            ref=np.max)


		tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
		Csync = librosa.util.sync(C, beats, aggregate=np.median)

		beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
		                                                            x_min=0,
		                                                            x_max=C.shape[1]),
		                                    sr=sr)

		R = librosa.segment.recurrence_matrix(Csync, width=1, mode='affinity',
		                                      sym=True)



		df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
		Rf = df(R, size=(1, 7))

		mfcc = librosa.feature.mfcc(y=y, sr=sr)
		Msync = librosa.util.sync(mfcc, beats)

		path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
		sigma = np.median(path_distance)
		path_sim = np.exp(-path_distance / sigma)

		R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)


		##########################################################
		# And compute the balanced combination (Equations 6, 7, 9)


		deg_path = np.sum(R_path, axis=1)
		deg_rec = np.sum(Rf, axis=1)
		Rf_bar = np.array([Rf[i][i%2+np.arange(0,len(deg_rec)-2,2)] for i in reversed(np.arange(0,len(deg_rec),2))])
		mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

		A = mu * Rf + (1 - mu) * R_path
		sum_A = np.sum(A,axis=0)

		#########################################

		song_t = song
		song_name = ''.join(song_t.split()[1:])[:-4]
		sum_Rf = np.sum(Rf,axis=0)
		sum_Rpath = np.sum(R_path,axis=0)
		sum_Rf_bar = np.sum(Rf_bar,axis=0)
		outstring = '_'.join(str(year).split())+' '+str(num)+' '+str(np.sum(sum_Rf))+' '+str(len(beat_times))+' '+str(max(beat_times))+' '+str(np.median(beat_times[1:]-beat_times[:-1]))+' '+\
			song_name+' '+str(np.sum(sum_Rpath))+' '+str(np.sum(sum_A))+' '+str(np.sum(sum_Rf_bar))+' '+str(mu)

		print outstring
		f.write(outstring)
		f.write('\n')

		'''
		plt.figure(figsize=(8, 8))
		librosa.display.specshow(Rf, cmap='gnuplot',y_axis='time', y_coords=beat_times, x_axis='time', x_coords=beat_times,vmin=0,vmax=1)
		plt.colorbar()
		plt.title(song)
		plt.tight_layout()
		plt.savefig(music_dir+'imgs/recurrence/'+str(year)+'_'+str(num)+'_'+song_name+'_r.png',dpi=150)



		plt.close()

		plt.figure(figsize=(8, 8))
		plt.plot(beat_times[:len(sum_Rf)],sum_Rf)
		

		plt.title(song)
		plt.xlabel('Time (s)')
		plt.ylabel('Sum of recurrence')
		plt.tight_layout()


		plt.savefig(music_dir+'imgs/collapsed/'+str(year)+'_'+str(num)+'_'+song_name+'_c.png',dpi=150)
		plt.close()
		'''
		#plt.show()

f.close()
