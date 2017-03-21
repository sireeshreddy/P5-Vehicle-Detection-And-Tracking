# P5-Vehicle-Detection-And-Tracking

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.jpg
[image2]: ./output_images/hog.jpg
[image3]: ./output_images/sliding_window.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.jpg
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell 5 of the IPython notebook. I used the code provided in the lessons.
```
# Define a function to return HOG features and visualization
# Vis == False means we do not want to get an image back, True produces output image.
def get_hog_features(img, 
                     orient, 
                     pix_per_cell, 
                     cell_per_block, 
                     vis=False, 
                     feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, 
                                  orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, 
                                  feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, 
                       orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, 
                       feature_vector=feature_vec)
        return features
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that the YUV and YCrCb colorspaces performed the best. Increasing the number of orientations and decreasing the number of pixels gave favorable results. These parameters were tweaked while training the classifier to identify the best fit.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, color bin and color histogram features in code cell 5. 
```
# Define feature parameters and test HOG classify using 500 car images and 500 non car images

color_space = 'YCrCb'
orient = 9
pix_per_cell = 6
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

t = time.time()
n_samples = 1000
# Generate 1000 random indices
random_idxs = np.random.randint(0 , len(cars), n_samples)
test_cars = np.array(cars)[random_idxs]
test_notcars = np.array(notcars)[random_idxs]

car_features = extract_features(test_cars,
                               color_space = color_space,
                               spatial_size = spatial_size,
                               hist_bins = hist_bins,
                               orient = orient,
                               pix_per_cell = pix_per_cell,
                               cell_per_block = cell_per_block,
                               hog_channel = hog_channel,
                               spatial_feat = spatial_feat,
                               hist_feat = hist_feat,
                               hog_feat = hog_feat
                               )

notcar_features = extract_features(test_notcars,
                               color_space = color_space,
                               spatial_size = spatial_size,
                               hist_bins = hist_bins,
                               orient = orient,
                               pix_per_cell = pix_per_cell,
                               cell_per_block = cell_per_block,
                               hog_channel = hog_channel,
                               spatial_feat = spatial_feat,
                               hist_feat = hist_feat,
                               hog_feat = hog_feat
                               )

print(time.time()-t, 'Seconds to compute features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X,
                                                   y,
                                                   test_size = 0.1,
                                                   random_state = rand_state
                                                   )
print('Using the following feature parameters ', '\n', color_space, 'Color Space', '\n', orient,' Orientations, ', '\n',
      pix_per_cell,'Pixels per cell ', '\n', cell_per_block,'Cells per block', '\n', hog_channel, 'HOG Channel', '\n', 
     spatial_size, 'Spatial Size', '\n', hist_bins, 'Hist Bins')
print('Feature vector length : ', len(X_train[0]))

# Use SVC
svc = LinearSVC()

t = time.time()
svc.fit(X_train, y_train)
print(round(time.time() -t, 2 ), "Seconds to train SVC...")

print('Test accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
```

My final model was a Linear SVM trained on:
YCrCb Color Space, 
9  Orientations, 
6 Pixels per cell, 
2 Cells per block, 
'ALL' HOG Channel, 
Spatial Size of (32, 32), 
Histogram features with 32 Bins. 
This resulted in a test accuracy of 99.5%. 

Some other settings resulted in higher test accuracy of 100% which I presumed was a sign of overfitting.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the sliding window search is implemented in code cell 8 which I obtained from the lessons. 
```
out_images = []
out_maps = []
out_titles = []
out_boxes = []
ystart = 400
ystop = 650
# Scale entire image and subsample the array
scale = 1.6

for img_src in test_images:
    img_boxes = []
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    
    # Make a heat map
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32) / 255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
   
    # Use // to remove floating points from results
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    
    nfeat_per_block = orient * cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            
            # Extract the image patch   
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
   
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart),(0,0,255),6)
                img_boxes.append(((xbox_left, ytop_draw +  ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                heatmap[ytop_draw + ystart:ytop_draw + win_draw + ystart, xbox_left:xbox_left + win_draw] +=1
                
    
    out_images.append(draw_img)
    
    out_titles.append('Test Image' + str(img_src[16:]))
    out_titles.append('Test Image' + str(img_src[16:]))
    
    
    out_images.append(heatmap)
    out_maps.append(heatmap)
    out_boxes.append(img_boxes)
#    mpimg.imsave('./output_images/heatmap' + str(img_src[16:]), heatmap)
#    mpimg.imsave('./output_images/bboxes' + str(img_src[16:]), draw_img)
    
fig = plt.figure(figsize = (12,24))
visualize(fig, 8, 2, out_images, out_titles) 
fig.savefig('./output_images/bboxes_and_heat.jpg')

```
I tried different scales ranging from 1 all the way till 2.5 and an overlap of 0.5 over the entire road surface 400 - 650 pixels. A scale of 1.6 gave optimal results.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To get an optimum result I used the feature extraction parameters described in the 'Training' section above. I used the YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images of the result I achieved:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/Crx-AdjaHcs)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I was able to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 
Although I defined a method to apply a threshold to the heatmap, I eventually ended up setting it to 0 and tweaked the scale instead. Placing a threshold helped minimize false positives but at a cost of not being capable of identifying vehicles further down the road.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
As is the case with any computer vision application, the biggest challenge is to identify the ideal parameters. Identifying the right colorspace and parameters was tricky as they did not always perform as expected. 
The YUV space gave similar results to the YCrCb colorspce while training. However, it did not perform as well on the final test video. Also, the train/test data was probably overfitting since the final video does not match the accuracy of 99.5% that was obtained upon testing of the trained model. 

One other avenue to speedup the pipeline as well as avoid false positives would be to isolate the region of interest to just the lanes in the direction of traffic and ignore cars on the other side of the freeway.

The biggest hurdle for the current pipeline is the computing time. It will have to process the frames in real-time if it has to have any real world application.
