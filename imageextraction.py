"""
Script to extract images from pytorch data set (in this case VOC).

This script shows all images in the VOC dataset one at a time.

Press q or Escape to exit,
press Enter or right arrow to continue to next image,
press s to extract the current image from the dataset and save it in data/extracted/,
press F11 to toggle fullscreen mode.
"""

import cv2
import os
import sys
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class FullScreenImageViewer:
    def __init__(self, image_loader):
        self.image_loader = image_loader
        self.image_loader_iter = iter(image_loader)
        self.tk_instance = tk.Tk()
        # self.tk_instance.attributes('-zoomed', True)
        self.tk_instance.attributes('-fullscreen', True)
        self.frame = tk.Frame(self.tk_instance)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.fullscreen_state = False
        self.tk_instance.bind("<F11>", self.toggle_fullscreen)
        self.tk_instance.bind("<Escape>", self.destroy)
        self.tk_instance.bind("q", self.destroy)
        self.tk_instance.bind("<Return>", self.next)
        self.tk_instance.bind("<KP_Enter>", self.next)
        self.tk_instance.bind("s", self.save_current_image)
        self.tk_instance.bind("<Right>", self.right_arrow)
        self.current_index = 0
        self.current_image = None
        self.show_current_image()
        self.toggle_fullscreen()
        self.frame.focus_force()

    def toggle_fullscreen(self, event=None):
        self.fullscreen_state = not self.fullscreen_state
        self.tk_instance.attributes("-fullscreen", self.fullscreen_state)
        return "break"

    def end_fullscreen(self, event=None):
        self.fullscreen_state = False
        self.tk_instance.attributes("-fullscreen", False)
        return "break"

    def next(self, event=None):
        if len(self.image_loader) - 1 <= self.current_index:
            self.destroy()
        else:
            self.current_index += 1
            self.show_current_image()
        return "break"

    def right_arrow(self, event=None):
        self.next(event)

    def destroy(self, event=None):
        self.tk_instance.destroy()

    def show_current_image(self):
        print("showing image {}".format(self.current_index))
        current_image = self.load_current_image()
        image_resized = self.resize_to_screen_size(current_image)
        self.render = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.render, anchor=tk.NW)

    def load_current_image(self):
        current_image = next(self.image_loader_iter)[0]
        current_image = current_image.numpy()[0]
        current_image = np.transpose(current_image, (1, 2, 0))
        current_image = (current_image * 255).astype(np.uint8)
        self.current_image = current_image
        return current_image

    def save_current_image(self, event=None):
        img_name = "Figure_{}.png".format(self.current_index)
        img_dir = os.path.join(os.getcwd(), "data/extracted")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        image_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, image_bgr)
        print("saved as {}".format(img_name))

    def resize_to_screen_size(self, image):
        # screen_width = self.tk_instance.winfo_screenwidth()
        # screen_height = self.tk_instance.winfo_screenheight()
        screen_width = 1920
        screen_height = 1080
        img_height, img_width, _ = image.shape

        factor = min(screen_height / img_height, screen_width / img_width)
        image_resized = cv2.resize(image, (0, 0), fx=factor, fy=factor)

        img_height, img_width, _ = image_resized.shape
        pad_height = max(0, (screen_height - img_height) // 2)
        pad_width = max(0, (screen_width - img_width) // 2)

        image_padded = cv2.copyMakeBorder(image_resized, top=pad_height, bottom=pad_height, left=pad_width,
                                          right=pad_width, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return image_padded

    def mainloop(self):
        self.tk_instance.mainloop()


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = torchvision.datasets.VOCDetection(root='./data', year='2012', image_set="train",
                                                 download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.VOCDetection(root='./data', year='2012', image_set="val",
                                                download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    viewer = FullScreenImageViewer(testloader)
    viewer.mainloop()
