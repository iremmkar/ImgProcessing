import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image 
import numpy as np
from skimage import color, data, exposure, filters, graph, io, morphology, segmentation, transform, util
import cv2
from skimage.segmentation import active_contour

def open_video():
    vid=cv2.VideoCapture(0)
    while True:
        ret,image=vid.read()
        cv2.rectangle(image,(100,100),(200,200),(25,36,98),3)
        cv2.line(image,(0,0),(100,100),(0,0,255),2)
        cv2.circle(image,(150,150),50,(80,150,35),2)
        cv2.putText(image,"Example Text",
            (250,250),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),3)
        cv2.imshow("video",image)
        if cv2.waitKey(25) & 0xFF==('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

panel=None
def openfn():
    filename =filedialog.askopenfilename(title='open')
    return filename
def open_img():

    global panel
    
    x = openfn()
    img = Image.open(x)
    image=np.asarray(img)
    im = img.resize((400, 250), Image.ANTIALIAS)
    im = ImageTk.PhotoImage(im)

    if len(x) > 0:
        def display(result):
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(image)
            ax[0].set_title("Original image")
            ax[1].imshow(result)
            ax[1].set_title("Result")
            plt.tight_layout()
            io.imshow(result)
            io.show()

        def save(I):
            io.imsave('c:\img.png',I)

        def f1():
            gray=color.rgb2gray(image)
            edge_sobel = filters.sobel(gray)
            display(edge_sobel)
        def f2():
            gray=color.rgb2gray(image)
            edge_roberts = filters.roberts(gray)
            display(edge_roberts)
        def f3():
            gray=color.rgb2gray(image)
            thresh = filters.threshold_mean(gray)
            binary = gray > thresh
            display(binary)
        def f4():
            gray=color.rgb2gray(image)
            filt_real, filt_imag = filters.gabor(gray, frequency=1)
            display(filt_real)
        def f5():
            gray=color.rgb2gray(image)
            mj=filters.meijering(gray, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0)
            display(mj)
        def f6():
            gray=color.rgb2gray(image)
            st=filters.sato(gray)
            display(st)
        def f7():
            gray=color.rgb2gray(image)
            fr=filters.frangi(gray, sigmas=range(1, 5, 2))
            display(fr)
        def f8():
            gray=color.rgb2gray(image)
            hsn=filters.hessian(gray)
            display(hsn)
        def f9():
            gsn = filters.gaussian(image, sigma=4, multichannel=True)
            display(gsn)
        def f10():
            gray=color.rgb2gray(image)
            ll=filters.laplace(gray)
            display(ll)

        #histogram equalization
        def histogram():
            I_eq = exposure.equalize_hist(image)
            display(I_eq)
        
        #transform
        def img_resize():
            rz =transform.resize(image, (image.shape[0] // 4, image.shape[1] // 4),anti_aliasing=True)
            display(rz)
        def warp_polar():
            wp=transform.warp_polar(image,output_shape=image.shape, scaling='log',multichannel=True)
            display(wp)
        def img_downscale():
            gray= color.rgb2gray(image)
            downscale = transform.downscale_local_mean(gray, (4,3))
            display(downscale)
        def img_rotate():
            rt=transform.rotate(image, 90, resize=True)
            display(rt)
        def img_swirl():
            sw=transform.swirl(image, rotation=0, strength=5, radius=180)
            display(sw)

        #rescale
        def img_rescale():

            #kullanıcıdan değer alma
            sayi=input("Bir sayi giriniz:")

            rc = transform.rescale(image, float(sayi), anti_aliasing=False,multichannel=True)
            display(rc)

        #Morphological filtering
        def img_mask():
            lum = color.rgb2gray(image)
            mask = morphology.remove_small_holes(morphology.remove_small_objects(lum < 0.7, 500),500)
            mask = morphology.opening(mask,morphology.disk(3))
            display(mask)
        def img_skeletonize():
            lum = color.rgb2gray(image)
            mask = morphology.remove_small_holes(morphology.remove_small_objects(lum < 0.7, 500),500)
            mask = morphology.opening(mask,morphology.disk(1))
            inverted_img = util.invert(image)
            skeleton = morphology.skeletonize(mask==0)
            display(skeleton)
        def img_watershed():
            gray=color.rgb2gray(image)
            edges = filters.sobel(gray)
            grid = util.regular_grid(gray.shape, n_points=468)
            seeds = np.zeros(gray.shape, dtype=int)
            seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
            w0 = morphology.watershed(edges, seeds)
            display(w0)
        def w_tophat():
            #threshold
            gray= color.rgb2gray(image) 
            thresh = filters.threshold_mean(gray)
            binary = gray > thresh

            org = util.img_as_ubyte(binary)
            ex = org.copy()
            ex[340:350, 200:210] = 255
            ex[100:110, 200:210] = 0
            selem = morphology.disk(6)
            wt=morphology.white_tophat(ex,selem)
            display(wt)
        def b_tophat():
            #threshold
            gray= color.rgb2gray(image) 
            thresh = filters.threshold_mean(gray)
            binary = gray > thresh

            org = util.img_as_ubyte(binary)
            ex = org.copy()
            ex[340:350, 200:210] = 255
            ex[100:110, 200:210] = 0
            selem = morphology.disk(6)
            bt=morphology.black_tophat(ex,selem)
            display(bt)
        def l_maxima():
            gray=color.rgb2gray(image)
            local_maxima = morphology.extrema.local_maxima(gray)
            label_maxima = morphology.label(local_maxima)
            overlay = color.label2rgb(label_maxima, gray, alpha=0.7, bg_label=0,bg_color=None, colors=[(1, 0, 0)])
            display(overlay)
        def img_erosion():
            #threshold
            gray= color.rgb2gray(image) 
            thresh = filters.threshold_mean(gray)
            binary = gray > thresh
            #erosion
            orig_img = util.img_as_ubyte(binary)
            selem = morphology.disk(6)
            esn= morphology.erosion(orig_img, selem)
            display(esn)

        def img_dilation():
            #threshold
            gray= color.rgb2gray(image) 
            thresh = filters.threshold_mean(gray)
            binary = gray > thresh
            #dilation
            orig_img = util.img_as_ubyte(binary)
            selem = morphology.disk(6)
            dilated = morphology.dilation(orig_img, selem)
            display(dilated)

        def img_opening():
            #threshold
            gray= color.rgb2gray(image) 
            thresh = filters.threshold_mean(gray)
            binary = gray > thresh
            #erosion
            orig_img = util.img_as_ubyte(binary)
            selem=morphology.disk(6)
            ersn = morphology.erosion(orig_img, selem)
            #opening
            opened = morphology.opening(ersn, selem)
            display(opened)

        def img_closing():
            #threshold
            gray= color.rgb2gray(image) 
            thresh = filters.threshold_mean(gray)
            binary = gray > thresh
            #dilation
            orig_img = util.img_as_ubyte(binary)
            selem = morphology.disk(6)
            dilated = morphology.dilation(orig_img, selem)
            #closing
            closed = morphology.closing(dilated, selem)
            display(closed) 

        def a_contour():      
            img=data.checkerboard()
            #img = color.rgb2gray(image)
            s = np.linspace(0, 2*np.pi, 400)
            r = 62 + 20*np.sin(s)
            c = 62 + 20*np.cos(s)
            init = np.array([r, c]).T
            snake = active_contour(filters.gaussian(img, 3),init, alpha=0.015, beta=10, gamma=0.001)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(img, cmap=plt.cm.gray)
            ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, img.shape[1], img.shape[0], 0])
            plt.show()

        if panel is None:
            panel = Label(root, image=im)
            panel.image = im
            panel.grid(column=0, row=2, columnspan=8)
        else:
            panel.configure(image=im)
            panel.image = im

        btn1 = Button(text= "Sobel",command=f1)
        btn1.grid(column=0, row=8,columnspan=1,sticky='nsew')
        btn2 = Button(text= "Roberts",command=f2)
        btn2.grid(column=1, row=8,columnspan=1,sticky='nsew')
        btn3 = Button(text= "Threshold",command=f3)
        btn3.grid(column=2, row=8,columnspan=1,sticky='nsew')
        btn4 = Button(text= "Gabor",command=f4)
        btn4.grid(column=3, row=8,columnspan=1,sticky='nsew')
        btn5 = Button(text= "Meijering",command=f5)
        btn5.grid(column=4, row=8,columnspan=1,sticky='nsew')
        btn6 = Button(text= "Sato",command=f6)
        btn6.grid(column=0, row=9,columnspan=1,sticky='nsew')
        btn7 = Button(text= "Frangi",command=f7)
        btn7.grid(column=1, row=9,columnspan=1,sticky='nsew')
        btn8 = Button(text= "Hessian",command=f8)
        btn8.grid(column=2, row=9,columnspan=1,sticky='nsew')
        btn9 = Button(text= "Gaussian",command=f9)
        btn9.grid(column=3, row=9,columnspan=1,sticky='nsew')
        btn10 = Button(text= "Laplace",command=f10)
        btn10.grid(column=4, row=9,columnspan=1,sticky='nsew')

        btn11 = Button(text = "Histogram equalization", command=histogram)
        btn11.grid(column=0, row=10,columnspan=3,sticky='nsew')

        btn12 = Button(text = "Resize", command=img_resize)
        btn12.grid(column=0, row=11,columnspan=1,sticky='nsew')
        btn13=Button(text="Warp polar",command=warp_polar)
        btn13.grid(column=1, row=11,columnspan=1,sticky='nsew')
        btn14=Button(text="Downscale", command=img_downscale)
        btn14.grid(column=2, row=11,columnspan=1,sticky='nsew')
        btn15=Button(text="Rotate",command=img_rotate)
        btn15.grid(column=3, row=11,columnspan=1,sticky='nsew')
        btn16=Button(text="Swirl",command=img_swirl)
        btn16.grid(column=4, row=11,columnspan=1,sticky='nsew')

        btn17=Button(text="Rescale", command=img_rescale)
        btn17.grid(column=0, row=12, columnspan=3, sticky='nsew')

        btn18=Button(text="Mask",command=img_mask)
        btn18.grid(column=0, row=13,columnspan=1,sticky='nsew')
        btn19=Button(text="Skeletonize",command=img_skeletonize)
        btn19.grid(column=1, row=13,columnspan=1,sticky='nsew')
        btn20=Button(text="Watershed", command=img_watershed)
        btn20.grid(column=2, row=13,columnspan=1,sticky='nsew')
        btn21=Button(text="White tophat", command=w_tophat)
        btn21.grid(column=3, row=13,columnspan=1,sticky='nsew')
        btn22=Button(text="Black tophat", command=b_tophat)
        btn22.grid(column=4, row=13,columnspan=1,sticky='nsew')

        btn23=Button(text="Local maxima", command=l_maxima)
        btn23.grid(column=0, row=14,columnspan=1,sticky='nsew')
        btn24=Button(text="Erosion", command=img_erosion)
        btn24.grid(column=1, row=14,columnspan=1,sticky='nsew')
        btn25=Button(text="Dilation", command=img_dilation)
        btn25.grid(column=2, row=14,columnspan=1,sticky='nsew')
        btn26=Button(text="Opening", command=img_opening)
        btn26.grid(column=3, row=14,columnspan=1,sticky='nsew')
        btn27=Button(text="Closing", command=img_closing)
        btn27.grid(column=4, row=14,columnspan=1,sticky='nsew')

        btn28=Button(text="Active contour", command=a_contour)
        btn28.grid(column=0, row=15,columnspan=3,sticky='nsew')

root=Tk()
root.resizable(width=True, height=True)
btn_vid=Button(root, text='Open Video',command=open_video).grid(column=1, row=0,columnspan=1,sticky='nsew')
btn_img = Button(root, text='Open Image', command=open_img).grid(column=0, row=0,columnspan=1,sticky='nsew')
root.mainloop()