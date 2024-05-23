import os
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter import simpledialog
import cv2
from skimage.morphology import skeletonize
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

WINDOW_HEIGHT = 600
WINDOW_WIDTH = 1280
RGB_CHANNELS = {"red": 0, "green": 1, "blue": 2}

mask = np.ones((5,5))
ksize_tuple = (5, 5)
ksize_value = 5
iterations = 1

history = []
image = None


stack_actions = False


def open_image(event = None):
    global image
    file_path = filedialog.askopenfilename()
    if file_path:
        #print(file_path)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display_image(image)

def display_image(img, undo_state=0):
    global history
    if undo_state == 0:
        history.append(img)
    pil_image = Image.fromarray(img)
    # Calculate the new size while preserving the aspect ratio
    ratio = min(WINDOW_WIDTH/pil_image.width, WINDOW_HEIGHT/pil_image.height)
    new_size = (int(pil_image.width*ratio), int(pil_image.height*ratio))
    # LANCZOS is a high-quality downsampling filter
    pil_image = pil_image.resize(new_size, resample=Image.LANCZOS)  
    photo_image = ImageTk.PhotoImage(pil_image)
    canvas.image = photo_image
    canvas.create_image((WINDOW_WIDTH/2, WINDOW_HEIGHT/2), image=photo_image, state="normal")

def save_image(event = None):
    global history
    
    if len(history) > 0:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")])
        _, extension = os.path.splitext(save_path)
        if save_path:
            try:
                converted_image = cv2.cvtColor(history[-1], cv2.COLOR_RGB2BGR)
                # Save the image in the correct format
                if extension == ".jpg":
                    quality = SliderDialog(window).result
                    print('Quality = ', quality)
                    if quality != None:
                        cv2.imwrite(save_path, converted_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        messagebox.showinfo("Success", "Image saved successfully!")
                        print('Saved!!')
                elif extension == ".png":
                    cv2.imwrite(save_path, converted_image)
                    messagebox.showinfo("Success", "Image saved successfully!")
                    print('Saved!!')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

def undo(event=None):
    global history
    if len(history) > 1:
        history.pop()
        display_image(history[-1], undo_state=1)

def clear_all(event=None):
    global image
    global def_image
    global history
    image = None
    history = []
    canvas.image = def_image
    canvas.create_image((WINDOW_WIDTH/2, WINDOW_HEIGHT/2), image=def_image, state="normal")


def rgb_to_hsv(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    display_image(converted_image)

def rgb_to_yuv(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    display_image(converted_image)

def rgb_to_hls(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    display_image(converted_image)

def rgb_to_lab(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    display_image(converted_image)

def rgb_to_luv(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
    display_image(converted_image)

def rgb_to_xyz(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    display_image(converted_image)

def rgb_to_ycrcb(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    display_image(converted_image)

def rgb_to_yuv(event=None):
    global image
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    display_image(converted_image)

def redish_image(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = local_img.copy()
    mask = np.zeros_like(c)
    mask[:,:] = [100,0,0]
    c = cv2.add(c, mask)   
    display_image(c)

def greenish_image(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = local_img.copy()
    mask = np.zeros_like(c)
    mask[:,:] = [0,100,0]
    c = cv2.add(c, mask)   
    display_image(c)

def blueish_image(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = local_img.copy()
    mask = np.zeros_like(c)
    mask[:,:] = [0,0,100]
    c = cv2.add(c, mask)   
    display_image(c)

def gray_image(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = cv2.cvtColor(local_img, cv2.COLOR_RGB2GRAY) 
    display_image(c)

def apply_compelement(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = 255 - local_img
    display_image(c)

def remove_channel(color, event=None):
    global image
    global RGB_CHANNELS
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    local_img[:,:, RGB_CHANNELS[color]] = 0
    display_image(local_img)

def naive_threshold(event=None):
    global image
    global RGB_CHANNELS
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    threshold = 127
    local_img[local_img <= threshold] = 0
    local_img[local_img > threshold] = 255
    display_image(local_img)

def adaptive_threshold(event=None):
    global image
    global RGB_CHANNELS
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    local_img = cv2.adaptiveThreshold(local_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    display_image(local_img)

def adaptive_threshold_mean(event=None):
    global image
    global RGB_CHANNELS
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    local_img = cv2.adaptiveThreshold(local_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    display_image(local_img)

def adaptive_threshold_otsu(event=None):
    global image
    global RGB_CHANNELS
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    _, local_img = cv2.threshold(local_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    display_image(local_img)

def adaptive_threshold_otsu_inv(event=None):
    global image
    global RGB_CHANNELS
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    _, local_img = cv2.threshold(local_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    display_image(local_img)

def rgb_dithering(event=None):
    global image
    global RGB_CHANNELS
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    local_img = local_img.astype(np.float64)  # Convert to float for calculations
    height, width, _ = local_img.shape
    for channel in range(3):  # Apply dithering separately to each color channel
        for y in range(height):
            for x in range(width):
                old_pixel = local_img[y, x, channel]
                new_pixel = round(old_pixel / 255) * 255
                local_img[y, x, channel] = new_pixel
                quant_error = old_pixel - new_pixel
                if x < width - 1:
                    local_img[y, x + 1, channel] += quant_error * 7 / 16
                if y < height - 1:
                    if x > 0:
                        local_img[y + 1, x - 1, channel] += quant_error * 3 / 16
                    local_img[y + 1, x, channel] += quant_error * 5 / 16
                    if x < width - 1:
                        local_img[y + 1, x + 1, channel] += quant_error * 1 / 16
    local_img = np.clip(local_img, 0, 255)  # Ensure all pixel intensities are within 0-255
    local_img = local_img.astype(np.uint8)  # Convert back to uint8 for display
    display_image(local_img)
    messagebox.showinfo("Info", "Dithering applied successfully!")

def gray_dithering(event=None):
    global image
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    local_img = local_img.astype(np.float64)  # Convert to float for calculations
    palette = np.array([0, 255], dtype=np.float64)
    height, width = local_img.shape
    for y in range(height):
        for x in range(width):
            old_pixel = local_img[y, x]
            new_pixel = min(palette, key=lambda p: abs(p - old_pixel))
            local_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x < width - 1:
                local_img[y, x + 1] += quant_error * 7 / 16
            if y < height - 1:
                if x > 0:
                    local_img[y + 1, x - 1] += quant_error * 3 / 16
                local_img[y + 1, x] += quant_error * 5 / 16
                if x < width - 1:
                    local_img[y + 1, x + 1] += quant_error * 1 / 16
    local_img = local_img.astype(np.uint8)  # Convert back to uint8 for display
    display_image(local_img)
    messagebox.showinfo("Info", "Dithering applied successfully!")


def rgb_hist_stretching(even=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    YCrC_image = cv2.cvtColor(local_img, cv2.COLOR_RGB2YCrCb)
    YCrC_image[:,:,0] = cv2.equalizeHist(YCrC_image[:,:,0])
    equalized_image = cv2.cvtColor(YCrC_image, cv2.COLOR_YCrCb2RGB)
    display_image(equalized_image)
    show_rgb_histogram(equalized_image)

def gray_hist_stretching(even=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = cv2.cvtColor(local_img, cv2.COLOR_RGB2GRAY)
    c = cv2.equalizeHist(c)
    display_image(c)

    hist_count = [0]*256
    for x in range(c.shape[0]):    
        for y in range(c.shape[1]):
            i = c[x,y]
            hist_count[i] = hist_count[i]+1 
    plt.plot(hist_count, color = "gray")
    plt.title("Gray Image' Histogram") 
    plt.show()


def show_histogram(even=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    show_rgb_histogram(local_img)

def show_rgb_histogram(rgb_image):
    max_val = []
    global RGB_CHANNELS
    for color in RGB_CHANNELS.keys():
        hist = cv2.calcHist(rgb_image, [RGB_CHANNELS[color]], None, [256], [0, 256]) 
        plt.subplot(2, 2, 1)
        plt.plot(hist, color = color)
        max_val.append(max(hist))
    
    plt.ylim(0, max(max_val) + 10)
    plt.title("RGB Image's Histograms")
    for idx, color in enumerate(RGB_CHANNELS.keys(), start=2):
        hist = cv2.calcHist(image, [RGB_CHANNELS[color]], None, [256], [0, 256]) 
        plt.subplot(2, 2, idx)
        plt.plot(hist, color = color)
        plt.title(f"{color.capitalize()} Image' Histogram")
        plt.ylim(0, max(max_val) + 10)  # Set the maximum y indication to be 255 + 10
    plt.subplots_adjust(hspace=0.5)  # Increase the padding between the plots
    plt.show()

def show_hist_for_gray(event=None):
    global image
    hist_count = [0]*256 
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    #a list of zeroes to store the frequency of all intensity values in the image (from 0 to 255) in.
    c = cv2.cvtColor(local_img, cv2.COLOR_RGB2GRAY)
    for x in range(c.shape[0]):    
        for y in range(c.shape[1]):
            i = c[x,y]
            hist_count[i] = hist_count[i]+1 
            #the intensity value is the index of its frequency in the list
    plt.plot(hist_count, color = "gray")
    plt.title("Gray Image' Histogram") 
    plt.show()


def apply_Max_Filter(event=None):
    global image
    global mask
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    max_filtered_image = cv2.dilate(local_img, mask)
    display_image(max_filtered_image)

def apply_Min_Filter(event=None):
    global image
    global mask
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    min_filtered_image = cv2.erode(local_img, mask, iterations=iterations)
    display_image(min_filtered_image)

def apply_Average_Filter(event=None):
    global image
    global mask
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    average = cv2.blur(local_img, ksize_tuple)
    display_image(average)

def apply_Range_Filter(event=None):
    global image
    global mask
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    range_filtered_image = cv2.subtract(cv2.dilate(local_img, mask, iterations=iterations), cv2.erode(local_img, mask, iterations=iterations))
    display_image(range_filtered_image)

def apply_Median_Filter(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    median = cv2.medianBlur(local_img , ksize_value)
    display_image(median)

def apply_Gaussian_Filter(event=None):
    global image
    global mask
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    gaus = cv2.GaussianBlur(local_img, ksize_tuple,30)
    display_image(gaus)

def apply_Laplacian_Filter(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    la = cv2.Laplacian(local_img,-1, ksize_tuple)
    display_image(la)

def opening(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = cv2.morphologyEx(local_img, cv2.MORPH_OPEN, mask, iterations=iterations)
    display_image(c)
    
def closing(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = cv2.morphologyEx(local_img, cv2.MORPH_CLOSE, mask, iterations=iterations)
    display_image(c)

def tophat(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = cv2.morphologyEx(local_img, cv2.MORPH_TOPHAT, mask, iterations=iterations)
    display_image(c)

def blackhat(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = cv2.morphologyEx(local_img, cv2.MORPH_BLACKHAT, mask, iterations=iterations)
    display_image(c)

def skeleton(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    c = skeletonize(local_img)
    display_image(c)

def change_brightness(event=None):
    global image
    local_img = []
    if stack_actions:
        local_img = history[-1]
    else:
        local_img = image
    brightness_factor = brightness_slider.get()
    c = local_img * brightness_factor
    c = np.clip(c, 0, 255)
    c = c.astype(np.uint8)
    display_image(c, undo_state=1)

def apply_brightness(event=None):
    brightness_label.config(text=f"Brightness Factor: {brightness_slider.get()}")
    change_brightness()

def switch():
    global stack_actions
     
    # Determine is on or off
    if stack_actions:
        on_button.config(image = off)
        stack_actions = False
    else:
        on_button.config(image = on)
        stack_actions = True
    print(f"Stack Actions is {stack_actions}")

def show_full_size_cv2(event=None):
    global history
    local_img = cv2.cvtColor(history[-1], cv2.COLOR_RGB2BGR)
    cv2.imshow("Full Size Image", local_img)

class SliderDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("JPEG Save Quality")
        self.slider = Scale(master, from_=0, to=100, orient="horizontal")
        self.slider.pack()
        return self.slider

    def buttonbox(self):
        box = Frame(self)

        x = Button(box, text="OK", width=10, command=self.apply, default=ACTIVE)
        x.pack(side=LEFT, padx=5, pady=5)

        w = Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=LEFT, padx=5, pady=5)

        self.bind("<Return>", self.apply)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def apply(self, event=None):
        self.result = self.slider.get()
        self.destroy()
    def cancel(self, event=None):
        self.result = None
        self.destroy()


window = Tk()
canvas = Canvas(window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, background='white')
def_image = PhotoImage(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
canvas.image = def_image
canvas.create_image((WINDOW_WIDTH/2, WINDOW_HEIGHT/2), image=def_image, state="normal")
#canvas.bind('<Button-1>', draw_line)
canvas.pack()



import_image = Button(window, text="Import an Image", command=open_image)
import_image.pack(side=LEFT, padx=10, pady=10)

save_button = Button(window, text="Save Image", command=save_image)
save_button.pack(side=LEFT, padx=10, pady=10)

full_size_button = Button(window, text="Open in full size", command=show_full_size_cv2)
full_size_button.pack(side=LEFT, padx=10, pady=10)

clear_button = Button(window, text="Clear All", command=clear_all)
clear_button.pack(side=LEFT, padx=10, pady=10)

undo_button = Button(window, text="Undo", command=undo)
undo_button.pack(side=LEFT, padx=10, pady=10)

brightness_slider = Scale(window, from_=0.0, to=5.0, orient=HORIZONTAL, resolution=0.1)
brightness_slider.set(1.0)
brightness_slider.pack(side=RIGHT, padx=10, pady=10)

brightness_label = Label(window, text=f"Brightness Factor: {brightness_slider.get()}")
brightness_label.pack(side=RIGHT, padx=10, pady=10)

brightness_slider.config(command=apply_brightness)

on = PhotoImage(file = "on.png")
off = PhotoImage(file = "off.png")

on_button = Button(window, image = off, bd = 0, command = switch)
on_button.pack(side=RIGHT, padx=10, pady=10)

toggle_label = Label(window, text="Stack Actions:")
toggle_label.pack(side=RIGHT, padx=0, pady=10)

# -------------------------MENU BAR SETTINGS--------------------------MENU BAR SETTINGS-------------------------------------------MENU BAR SETTINGS-----------

menu_bar = Menu(window)
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="New", accelerator="Ctrl+N", command=open_image)
file_menu.add_command(label="Save", accelerator="Ctrl+S", command=save_image)

filters_menu = Menu(menu_bar, tearoff=0)
filters_menu.add_command(label="Max Filter", command=apply_Max_Filter)
filters_menu.add_command(label="Min Filter", command=apply_Min_Filter)
filters_menu.add_command(label="Average Filter", command=apply_Average_Filter)
filters_menu.add_command(label="Range Filter", command=apply_Range_Filter)
filters_menu.add_command(label="Median Filter", command=apply_Median_Filter)
filters_menu.add_command(label="Gaussian Filter", command=apply_Gaussian_Filter)
filters_menu.add_command(label="Laplacian Filter", command=apply_Laplacian_Filter)
filters_menu.add_command(label="Opening", command=opening)
filters_menu.add_command(label="Closing", command=closing)
filters_menu.add_command(label="Top Hat", command=tophat)
filters_menu.add_command(label="Black Hat", command=blackhat)
filters_menu.add_command(label="Skeletonize", command=skeleton)
filters_menu.add_command(label="RGB Dithering", command=rgb_dithering)
filters_menu.add_command(label="Gray Dithering", command=gray_dithering)

histogram_menu = Menu(menu_bar, tearoff=0)
histogram_menu.add_command(label="RGB Histogram", accelerator="Ctrl+1", command=show_histogram)
histogram_menu.add_command(label="Gray Histogram", accelerator="Ctrl+2", command=show_hist_for_gray)
histogram_menu.add_command(label="RGB Histogram Stretching", accelerator="Ctrl+3", command=rgb_hist_stretching)
histogram_menu.add_command(label="GRAY Histogram Stretching", accelerator="Ctrl+4", command=gray_hist_stretching)

color_manipulation_menu = Menu(menu_bar, tearoff=0)
color_manipulation_menu.add_command(label="Redish Image", accelerator="Ctrl+R", command=redish_image)
color_manipulation_menu.add_command(label="Greenish Image", accelerator="Ctrl+G", command=greenish_image)
color_manipulation_menu.add_command(label="Blueish Image", accelerator="Ctrl+B", command=blueish_image)
color_manipulation_menu.add_command(label="Gray Image", accelerator="Ctrl+W", command=gray_image)
color_manipulation_menu.add_command(label="Complement Image", accelerator="Ctrl+C", command=apply_compelement)
color_manipulation_menu.add_command(label="Remove Red Color", accelerator="Shift+R", command=lambda: remove_channel("red"))
color_manipulation_menu.add_command(label="Remove Green Color", accelerator="Shift+G", command=lambda: remove_channel("green"))
color_manipulation_menu.add_command(label="Remove Blue Color", accelerator="Shift+B", command=lambda: remove_channel("blue"))
color_manipulation_menu.add_command(label="Naive Threshold", command=naive_threshold)
color_manipulation_menu.add_command(label="Adaptive Threshold", command=adaptive_threshold)
color_manipulation_menu.add_command(label="Adaptive Threshold (Mean)", command=adaptive_threshold_mean)
color_manipulation_menu.add_command(label="Adaptive Threshold (Otsu)", command=adaptive_threshold_otsu)
color_manipulation_menu.add_command(label="Adaptive Threshold (Otsu Inverted)", command=adaptive_threshold_otsu_inv)

color_spaces_menu = Menu(menu_bar, tearoff=0)
color_spaces_menu.add_command(label="RGB to HSV", command=rgb_to_hsv)
color_spaces_menu.add_command(label="RGB to YUV", command=rgb_to_yuv)
color_spaces_menu.add_command(label="RGB to HLS", command=rgb_to_hls)
color_spaces_menu.add_command(label="RGB to LAB", command=rgb_to_lab)
color_spaces_menu.add_command(label="RGB to LUV", command=rgb_to_luv)
color_spaces_menu.add_command(label="RGB to XYZ", command=rgb_to_xyz)
color_spaces_menu.add_command(label="RGB to YCrCb", command=rgb_to_ycrcb)
color_spaces_menu.add_command(label="RGB to YUV", command=rgb_to_yuv)

menu_bar.add_cascade(menu=file_menu, label="File")
menu_bar.add_cascade(menu=filters_menu, label="Filters")
menu_bar.add_cascade(menu=histogram_menu, label="Histograms")
menu_bar.add_cascade(menu=color_manipulation_menu, label="Color Manipulation")
menu_bar.add_cascade(menu=color_spaces_menu, label="Color Spaces")

# -------------------------KEY BINDING SETTINGS--------------------------KEY BINDING SETTINGS-------------------------------------------KEY BINDING SETTINGS-----------

window.bind_all("<Control-z>", undo)
window.bind_all("<Control-Z>", undo)
window.bind_all("<Control-n>", open_image)
window.bind_all("<Control-N>", open_image)
window.bind_all("<Control-s>", save_image)
window.bind_all("<Control-S>", save_image)
window.bind_all("<Control-r>", redish_image)
window.bind_all("<Control-R>", redish_image)
window.bind_all("<Control-g>", greenish_image)
window.bind_all("<Control-G>", greenish_image)
window.bind_all("<Control-b>", blueish_image)
window.bind_all("<Control-B>", blueish_image)
window.bind_all("<Shift-r>", lambda event:remove_channel("red"))
window.bind_all("<Shift-R>", lambda event:remove_channel("red"))
window.bind_all("<Shift-g>", lambda event:remove_channel("green"))
window.bind_all("<Shift-G>", lambda event:remove_channel("green"))
window.bind_all("<Shift-b>", lambda event:remove_channel("blue"))
window.bind_all("<Shift-B>", lambda event:remove_channel("blue"))
window.bind_all("<Control-w>", gray_image)
window.bind_all("<Control-W>", gray_image)
window.bind_all("<Control-c>", apply_compelement)
window.bind_all("<Control-C>", apply_compelement)
window.bind_all("<Control-Key-1>", show_histogram)
window.bind_all("<Control-Key-2>", show_hist_for_gray)
window.bind_all("<Control-Key-3>", rgb_hist_stretching)
window.bind_all("<Control-Key-4>", gray_hist_stretching)

# -------------------------WINDOWS CONFIGURATION------------------WINDOWS CONFIGURATION----------------------------WINDOWS CONFIGURATION---------------------

window.title("Mini Photoshop")
window.resizable(False, False)
window.configure(bg='lightblue', menu=menu_bar)
#window.config(menu=menu_bar)

print("Made by Zeyad Hemeda\n")
window.mainloop()
