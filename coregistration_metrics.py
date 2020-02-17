from matplotlib import pyplot as plt
from imzmlparser import _bisect_spectrum
from scipy.stats import pearsonr
from mykmeans import MiniBatchKMeans
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN
import random

# transform image given X shift, Y shift, angle of rotation (radians) and scaling factor
def transform_image(X,theta,shiftX,shiftY,scale): 
    (height, width) = X.shape[:2]
    d = (height*height*0.25+width*width*0.25)**0.5
    beta = np.arctan(height/width)
    pad_size = np.abs(int(d*np.cos(beta - theta)-width/2))
    vertical_pad = np.zeros((height,pad_size))
    horizontal_pad = np.zeros((pad_size,2*pad_size+width))
    X_pad = np.concatenate((vertical_pad,X,vertical_pad),axis = 1)
    X_pad2 = np.concatenate((horizontal_pad,X_pad,horizontal_pad),axis = 0)
    (height2, width2) = X_pad2.shape[:2]
    # rotation
    M = cv2.getRotationMatrix2D((width2/2,height2/2), np.degrees(theta), 1)
    Y = cv2.warpAffine(X_pad2, M, (width2,height2))
    # scaling
    Y = cv2.resize(Y, dsize=(int(width2*scale),int(height2*scale)))
    (height3, width3) = Y.shape[:2]
    padY = int((height3-height)/2)
    padX = int((width3-width)/2)
    # shift
    if height3 < height and width3<width:
        output = np.zeros((height, width))
        output[-padY:-padY+height3,-padX:-padX+width3] = Y
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        res = cv2.warpAffine(output, M, (width,height))
        return res
    elif height3 > height and width3 > width:
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        res = cv2.warpAffine(Y, M, (width3,height3))
        return res[padY:padY+height,padX:padX+width]
    else:
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        res = cv2.warpAffine(Y, M, (width3,height3))
        return cv2.resize(res, dsize=(width,height))

# alignment metric based on image correlation. The bigger mutual information of two images the better is the alignment
def mutual_information(p,X,Y):
    (height, width) = X.shape[:2]
    if int(p[3]*height)*int(p[3]*width) == 0: return 0
    transformed_Y = transform_image(Y,p[0],p[1],p[2],p[3])
    # add a fine factor to account for the parts of the image that were placed outside of the initial field of view
    fine = np.sum(transformed_Y > 0)/np.sum(Y > 0)
    X = X.ravel()
    transformed_Y = transformed_Y.ravel()
    # not comparing regions that have zeros in them - actual MALDI and confocal background always > 0
    X = X[np.where(transformed_Y != 0)]
    transformed_Y = transformed_Y[np.where(transformed_Y != 0)]
    transformed_Y = transformed_Y[np.where(X != 0)]
    X = X[np.where(X != 0)]
    # calculating mutual information
    hgram, _, _ = np.histogram2d(X,transformed_Y,bins=20)
    pxy = hgram / float(np.sum(hgram))
    py = np.sum(pxy, axis=1) 
    px = np.sum(pxy, axis=0) 
    px_py = py[:, None] * px[None, :] 
    nzs = pxy > 0 
    # adding the minus to the output value to use this metric for minimization function 
    # actually looking for the global maximum of mutual information in the field of all possible transformations from the
    # transform_image function
    return -fine*fine*np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def crop_zeros(image):
    # crop out stripes of zeros on the sides of the image
    shape = image.shape
    top,bottom,left,right = 0,shape[0],0,shape[1]
    for i in range(shape[0]):
        s = np.sum(image[i,:])
        if s > 0:
            bottom = i
            if top == 0: top = i
    for i in range(shape[1]):
        s = np.sum(image[:,i])
        if s > 0:
            right = i
            if left == 0: left = i  
    # crop out zeros in corners that appeared due to rotation
    delta_left, delta_right, delta_top, delta_bottom = 0,0,0,0
    while image[top,left+delta_left] == 0: delta_left += 1
    while image[top+delta_top,left] == 0: delta_top += 1
    if delta_top < delta_left: top += delta_top
    else: left += delta_left
    delta_left, delta_top = 0,0 
    while image[bottom,left+delta_left] == 0: delta_left += 1
    while image[bottom+delta_bottom,left] == 0: delta_bottom -= 1        
    if -delta_bottom < delta_left: bottom += delta_bottom
    else: left += delta_left  
    delta_bottom = 0    
    while image[bottom,right+delta_right] == 0: delta_right -= 1
    while image[bottom+delta_bottom, right] == 0: delta_bottom -= 1        
    if delta_bottom > delta_right: bottom += delta_bottom
    else: right += delta_right 
    delta_right = 0
    while image[top,right+delta_right] == 0: delta_right -= 1
    while image[top+delta_top,right] == 0: delta_top += 1        
    if delta_top < -delta_right: top += delta_top
    else: right += delta_right   
    # return crop boundaries    
    return [top,bottom,left,right]
         
# remove the cells with area outside the 3*sigma range, producing the cell colored by area plot
def cleanup(cellprops, h, w, output_folder):
    n = len(cellprops)
    area = np.zeros(n)
    for i in range(n): 
        area[i] = cellprops[i]["area"]
    area_std, area_mean = np.std(area), np.mean(area)
    output_area, output_perimeter, output_eccentricity = np.zeros((h,w)), np.zeros((h,w)), np.zeros((h,w))
    cells, X, Y, area, perimeter, eccentricity = [],[],[],[],[],[]
    for i in range(n):
        if cellprops[i]['area'] < area_mean+3*area_std and cellprops[i]['area'] > max(1,area_mean-3*area_std):
            cells.append(cellprops[i])
            X.append(cellprops[i]['centroid'][0])
            Y.append(cellprops[i]['centroid'][1])
            area.append(cellprops[i]["area"])
            perimeter.append(cellprops[i]["perimeter"])
            eccentricity.append(cellprops[i]["eccentricity"])
            for coord in cellprops[i]["coords"]:
                output_area[coord[0],coord[1]] = cellprops[i]['area'] 
                output_perimeter[coord[0],coord[1]] = cellprops[i]['perimeter'] 
                output_eccentricity[coord[0],coord[1]] = cellprops[i]['eccentricity'] 
    plot(output_area,output_folder+"/Area.png")
    plot(output_perimeter,output_folder+"/Perimeter.png")
    plot(output_eccentricity,output_folder+"/Eccentricity.png")
    return cells, np.array(X), np.array(Y), area, perimeter, eccentricity

def plot(image,name):
    m = np.max(image)
    cmap = colors.ListedColormap(['white','darkblue','blue','cornflowerblue',
                                  'cyan','aquamarine','lime','greenyellow','yellow','gold','orange','red','brown'])
    bounds=[0,0.00001,m/12,m/6,m/4,m/3,m/2.4,m/2,0.58*m,m/1.5,0.75*m,m/1.2,0.91*m,m]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    image = plt.imshow(image, cmap=cmap, norm=norm)
    plt.colorbar(image, cmap=cmap, norm=norm, boundaries=bounds, ticks=np.around(bounds, decimals=1))
    plt.savefig(name,dpi=1200,bbox_inches = "tight")
    plt.close()

# overlay cells segmented from the confocal image on the MALDI images, and get the corresponding m/z intensities on 
# per cell basis
def overlay_cells(image, cells, name):
    h,w = image.shape
    output = np.zeros((h,w))
    n = len(cells)
    intensities = np.zeros(n)
    for i in range(n):
        for coord in cells[i]["coords"]:
            intensities[i] += image[coord[0],coord[1]]
        intensities[i] = intensities[i]/cells[i]["area"]
        for coord in cells[i]["coords"]:
            output[coord[0],coord[1]] = intensities[i]
    plot(output,name)
    return intensities

# visualize metrics, saving in the output folder provided through the "name" variable
def visualize(metric, cells, h, w, name):
    output = np.zeros((h,w))
    n = len(cells)
    for i in range(n):
        for coord in cells[i]["coords"]:
            output[coord[0],coord[1]] = metric[i]
    plot(output,name)

# get average distance between cells' centers from a random rectangular sample in the image
def get_connection_length(X,Y):
    left = random.randrange(int(np.min(X)), int(0.75*np.max(X)))
    top = random.randrange(int(np.min(Y)), int(0.75*np.max(Y)))
    rand_ind = np.where(np.logical_and(np.logical_and(X > left,X < (left + 0.25*np.max(X))), np.logical_and(Y > top, Y < (top + 0.25*np.max(Y)))))
    X = X[rand_ind]
    Y = Y[rand_ind]
    n = len(X)
    connections = np.zeros(n)
    for i in range(n):
        d = np.max(X)
        for j in range(n):
            if i == j: continue
            dist = ((X[j]-X[i])**2+(Y[j]-Y[i])**2)**0.5
            if dist < d: d = dist
        connections[i] = d
    mu = np.mean(connections)
    sigma = 3*np.std(connections)
    main_body = np.logical_and(connections < (mu+sigma), connections > (mu-sigma))
    connections = connections[main_body]
    l = np.max(connections)
    return l

# identify if the cell is on the edge of the colony
def on_edge(X,Y,i,count,neigh,margin):
    selfX = X[i]
    selfY = Y[i]
    # cells closer to the edge of the image than the given margin are considered "on-colony"
    if selfX < margin or selfY < margin or selfX > max(X)-margin or selfY > max(Y)-margin: return 0
    # output 0 means cell is off-edge, 1 - on the edge
    for j in range(count):
        neighX = X[neigh[j]]
        neighY = Y[neigh[j]]
        if selfX == neighX:
            flag = 1
        else:
            a = (selfY - neighY) / (selfX-neighX)
            b = selfY - a * selfX
            flag = 0
        s = 0
        for k in range(count):
            if k == j: continue
            if flag == 0:
                s = s + np.sign(Y[neigh[k]]-a*X[neigh[k]]-b) 
            else:
                s = s + np.sign(Y[neigh[k]] - selfY)
        if np.abs(s) == count: return 1
        else: edge = 0
            
    return edge

# calculating distance from the edge of the colony for a given cell
def edgeDistance(X,Y,edge,i):
    distances = ((X[np.where(edge==1)]-X[i])**2+(Y[np.where(edge==1)]-Y[i])**2)**0.5
    return np.min(distances)

def ClusterImage(Z,k,X,Y,epsilon,minclust):
    if np.sum(Z) == 0: return np.zeros(len(Z)), np.ones(len(Z))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Z.reshape(-1, 1))
    idx = kmeans.labels_
    l = len(Z)
    cluster_sizes = np.zeros(l)
    cluster_avg = np.zeros(l)
    nums = np.arange(0,l)
    randomness = 0
    for i in range(k):
        indices = nums[np.where(idx == i)]
        x = X[indices]
        y = Y[indices]
        z = Z[indices]
        coeffs = nums[indices]
        clustering = DBSCAN(eps=epsilon, min_samples=minclust).fit(np.array([x,y]).reshape(-1, 2))
        idx2 = clustering.labels_
        dump = 0
        sizes = []
        avgs = []
        for j in range(len(idx2)):
            if idx2[j] == -1:
                sizes.append(1)
                if z[j] == 0:
                    avgs.append(0.01)
                else:
                    avgs.append(z[j])
                dump += 1
            else:
                sizes.append(np.sum(idx2 == idx2[j])) 
                if np.sum(z[np.where(idx2 == idx2[j])]) == 0:
                    avgs.append(0.01)
                else:
                    avgs.append(np.sum(z[np.where(idx2 == idx2[j])])/np.sum(idx2 == idx2[j]))
        cluster_sizes[coeffs] = sizes
        cluster_avg[coeffs] = avgs
        randomness = randomness + np.max(idx2) + 2*dump
        nums[np.where(idx == i)] = idx2+np.max(nums)*np.ones(len(idx2))-np.array(idx2==0,dtype = int)*np.max(nums)
    
    randomness = randomness/len(Z)

    return cluster_sizes, cluster_avg

def getmaxmin(p):
    xmax, xmin, ymax, ymin = 0,600000,0,600000
    for i, (x, y, z) in enumerate(p.coordinates):
        if x>xmax:xmax = x
        if x<xmin:xmin = x
        if y>ymax:ymax = y
        if y<ymin:ymin = y
    return xmax, xmin, ymax, ymin

def record_reader(borders,p,MALDI_output,mz_values,tolerances):
    print("Creating",len(mz_values),"m/z images")
    cmap = plt.cm.gray
    img,max_int,sum_im,_,_= getionimage2(borders,p, mz_values, tolerances, z=1, reduce_func=sum,dim=len(mz_values))
    max_sum_im = np.max(sum_im)
    print("Average signal over provided peaks:")
    plt.imshow(sum_im/max_sum_im,cmap="gray")
    plt.colorbar()
    plt.show()
    plt.imsave(MALDI_output+"//average.png", cmap(sum_im/max_sum_im))
    print("Maximum signal value:",max_int)
    for index,values in enumerate(img):
        values =(values)/(max_int)
        a = cmap(values)
        exten = str(mz_values[index])
        exten.replace(".","_")
        plt.imsave("{}//MALDI__{}.png".format(MALDI_output,exten), a)
        
    
def correlation_segmentation_reference(p,colony,xmin,ymin,xmax,ymax,output_folder,x_ref = -1,y_ref = -1,all = True):
    imcorr = np.zeros((ymax-ymin, xmax-xmin))
    pearsons = []
    xs = []
    ys = []
    x = random.randint(xmin, xmax)
    y = random.randint(ymin, ymax)
    print(x,y)
    while colony[y-ymin-1, x-xmin-1] != 255:
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        print(x,y)
        
    if x_ref == -1:
        for i, (x_, y_, z_) in enumerate(p.coordinates):
            if x == x_ and y == y_:
                _, spectr = map(lambda temp: np.asarray(temp), p.getspectrum(i))
                break
    else:
        for i, (x, y, z) in enumerate(p.coordinates):
            if x == x_ref and y == y_ref:
                _, spectr = map(lambda temp: np.asarray(temp), p.getspectrum(i))
                break
        if i == len(p.coordinates): print("Given reference point is not found: using random on-colony point as reference")

    for idx, (x1, y1, z1) in enumerate(p.coordinates):
        if x1 > xmin and y1 > ymin and x1 < xmax and y1 < ymax:
            if all or colony[y1-ymin-1, x1-xmin-1] > 0:
                _, spectr2 = map(lambda temp: np.asarray(temp), p.getspectrum(idx))
                corr, _ = pearsonr(spectr, spectr2)
                pearsons.append([corr])
                imcorr[y1-ymin-1,x1-xmin-1] = corr
                xs.append(x1-xmin-1)
                ys.append(y1-ymin-1)

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0,xmax-xmin,10))
    ax.set_xticklabels(np.arange(xmin,xmax,10))
    ax.set_yticks(np.arange(0,ymax-ymin,10))
    ax.set_yticklabels(np.arange(ymin,ymax,10))
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.imshow(imcorr,cmap ='jet')
    plt.colorbar()
    plt.show()
    cmap = plt.cm.jet
    a = cmap(imcorr)
    plt.imsave(output_folder+"/correlation heatmap x_ref"+str(x)+"y_ref"+str(y)+".png",a)



def kmeanscorrelationfull(p,colony,xmin,ymin,xmax,ymax,k,output_folder,all = True):
    marray,l = [],np.array([])
    coordsX = []
    coordsY = []
    im = np.zeros((ymax-ymin+1, xmax-xmin+1))
    f = open(output_folder+"/kmeans "+str(k)+" output.txt","w")
    kmeans = MiniBatchKMeans(n_clusters=k,random_state=0,batch_size=20000)
    for idx, (x1, y1, z1) in enumerate(p.coordinates):
        if x1 > xmin and y1 > ymin and x1 < xmax and y1 < ymax:
            if all or colony[y1-ymin-1, x1-xmin-1] > 0:
                mzs, ints = p.getspectrum(idx)        
                marray.append(ints)
                coordsX.append(x1-xmin-1)
                coordsY.append(y1-ymin-1)
                if len(marray) % 20000 == 0:
                    kmeans = kmeans.partial_fit(marray)
                    l = np.concatenate((l,kmeans.labels_))
                    print("done")
                    marray = []
                    for i in range(len(coordsX)):
                        im[coordsY[i],coordsX[i]] = l[i]+1
                    plt.imshow(im,cmap ='jet')
                    plt.colorbar()
                    plt.show()
    print("starting kmeans last")
    kmeans = kmeans.partial_fit(marray)
    l = np.concatenate((l,kmeans.labels_))
    c = kmeans.cluster_centers_
    print("done")     
    f_c = open(output_folder+"/kmeans "+str(k)+" centers.txt","w")
    for i in range(len(mzs)):
        f_c.write(str(mzs[i])+" ")
        for j in range(k):
            f_c.write(str(c[j][i])+" ")
        f_c.write("\n")
    f_c.close()
    for i in range(len(coordsX)):
        im[coordsY[i],coordsX[i]] = l[i]+1
        f.write(str(coordsY[i])+" "+str(coordsX[i])+" "+str(l[i])+"\n")
    f.close()
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0,xmax-xmin,10))
    ax.set_xticklabels(np.arange(xmin,xmax,10))
    ax.set_yticks(np.arange(0,ymax-ymin,10))
    ax.set_yticklabels(np.arange(ymin,ymax,10))
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.imshow(im,cmap ='jet')
    plt.colorbar()
    plt.show()
    cmap = plt.cm.jet
    a = cmap(im/k)
    plt.imsave(output_folder+"/kmeans "+str(k)+" output.png",a)
    return l
    

def getionimage2(borders,p, mz_values, tolerances, z=1, reduce_func=sum,dim=1,):
    coordsX = []
    coordsY = []
    max_int = 0.0
    im = np.zeros((dim, borders[2]-borders[1]+1,  borders[3]-borders[0]+1))
    sum_im = np.zeros((borders[2]-borders[1]+1,  borders[3]-borders[0]+1))
    for i, (x, y, z_) in enumerate(p.coordinates):
        if z_ == z and x > borders[0] and y > borders[1] and x < borders[3] and y < borders[2]:
            mzs, ints = p.getspectrum(i)
            coordsX.append(x-borders[0]-1)
            coordsY.append(y-borders[1]-1)
            for index,mz_value in enumerate(mz_values):
                min_i, max_i = _bisect_spectrum(mzs, mz_value, tolerances[index])
                im[index,y-borders[1]-1, x-borders[0]-1] = reduce_func(ints[min_i:max_i+1])
                sum_im[y-borders[1]-1, x-borders[0]-1] += reduce_func(ints[min_i:max_i+1])
                if im[index,y-borders[1]-1, x-borders[0]-1] > max_int:
                    max_int = im[index,y-borders[1]-1, x-borders[0]-1]
    return im, max_int, sum_im, coordsX, coordsY

    
