from Kinho import Neural
from PIL import Image
import numpy as np
import random

RESIZED_WIDTH = 30
RESIZED_WEIGHT = 30

def isInsideOnEllipse(ellipse, x, y):
    major_axis_radius = ellipse['major_axis_radius']
    minor_axis_radius = ellipse['minor_axis_radius']
    angle = ellipse['angle']
    center_x = ellipse['center_x']
    center_y = ellipse['center_y']
    
    x_translated = x - center_x
    y_translated = y - center_y
    
    x_rotated = x_translated * np.cos(-angle) - y_translated * np.sin(-angle)
    y_rotated = x_translated * np.sin(-angle) + y_translated * np.cos(-angle)

    coef = ((x_rotated / major_axis_radius) ** 2 + (y_rotated / minor_axis_radius) ** 2)

    return coef <= 1.0

def buildDataset():
    imgs = []
    
    for i in range(1, 11):
        url = 'data/face/FDDB-folds/FDDB-fold-'
        if i < 10:
            url += '0'
        url += str(i) + '-ellipseList.txt'
        
        f = open(url)
        img_url = f.readline()[:-1]
        while img_url != "":        
            n_ellipse = int(f.readline())
            ellipses = []
            
            for _ in range(n_ellipse):
                vals = f.readline().split(" ")
                ellipses.append({
                    'major_axis_radius': float(vals[0]),
                    'minor_axis_radius': float(vals[1]),
                    'angle': float(vals[2]),
                    'center_x': float(vals[3]),
                    'center_y': float(vals[4])
                })
            
            imgs.append({
                'path': img_url,
                'ellipses': ellipses
            })
            
            img_url = f.readline()[:-1]
        f.close()

    return imgs

MAX_PIXELS = 0

def maskImages(info):
    global MAX_PIXELS

    tot = len(info)
    i = 0
    for data in info:
        img = Image.open('data/face/' + data['path'] + '.jpg')
        img_copy = Image.open('data/face/' + data['path'] + '.jpg')
        width, height = img.size
        
        for x in range(width):
            for y in range(height):
                isInside = False
                for ell in data['ellipses']:
                    if isInsideOnEllipse(ell, x, y):
                        isInside = True
                        break
                
                color = 255 if isInside else 0
                img_copy.putpixel((int(x), int(y)), color)
        
        img = img.resize((RESIZED_WIDTH, RESIZED_WEIGHT))
        img_copy = img.resize((RESIZED_WIDTH, RESIZED_WEIGHT))

        gray_img = img.convert('L')
        gray_img_copy = img_copy.convert('L')
        pixels = gray_img.load()
        pixels_copy = gray_img_copy.load()
        img_mask = []
        label_mask = []
        
        MAX_PIXELS = max(MAX_PIXELS, RESIZED_WIDTH * RESIZED_WEIGHT)
        
        for x in range(RESIZED_WIDTH):
            for y in range(RESIZED_WEIGHT):
                img_mask.append(float(pixels[x, y] / 255))
                
                isInside = pixels_copy[x, y] < 190
                
                label_mask.append(1.0 if isInside else 0.0)
        
        data['mask'] = {
            'img': img_mask,
            'label': label_mask
        }
        
        if i%100 == 0:
            print("{}/{} - {}".format(i//100, tot//100, MAX_PIXELS))
        i += 1
    
    return info

def main():
    dataset = buildDataset()
    train = maskImages(dataset[:100])
    test = maskImages(dataset[-100:])
    
    print("MAXPIXELS = {}".format(MAX_PIXELS))
    
    bot = Neural(
        sizes=[RESIZED_WEIGHT * RESIZED_WIDTH, 50, 50, RESIZED_WIDTH * RESIZED_WEIGHT],
        eta=0.1,
        gpu=True,
        mini_batch_size=32,
        multilabel=True
    )
    
    PERIOD = 200
    
    for i in range(100):
        score = 0
        min_cost = 100
        avg_cost = 0
        for img in test:
            c = bot.cost(img['mask']['img'], img['mask']['label'])
            if c < 0.1:
                score += 1
            min_cost = min(min_cost, c)
            avg_cost += c
        
        print("EPOCH = {}\nSCORE = {}\nMin_COST = {}\nAvg_COST = {}".format(
            i,score/len(test), min_cost, avg_cost/len(test)
        ))
        
        j = 1
        for img in train:
            bot.learn(img['mask']['img'], img['mask']['label'])
            if j%PERIOD == 0:
                print("{}/{}".format(j//PERIOD, len(train)//PERIOD))
            j += 1

if __name__ == "__main__":
    main()
