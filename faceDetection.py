from Kinho import Neural
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

RESIZED_WIDTH = 60
RESIZED_WEIGHT = 60

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
    aleat = True

    tot = len(info)
    cnt = 0
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
        
        if random.randint(1, 800) == 7:
            aleat = True
        
        if aleat:
            plt.figure(figsize=(10, 3))

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img_copy)
            plt.axis('off')

            plt.savefig("test.png")
        
        img = img.resize((RESIZED_WIDTH, RESIZED_WEIGHT))
        img_copy = img_copy.resize((RESIZED_WIDTH, RESIZED_WEIGHT))
        
        if aleat:
            plt.figure(figsize=(10, 3))

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(img_copy)
            plt.axis('off')

            plt.savefig("test2.png")

        gray_img = img.convert('L')
        gray_img_copy = img_copy.convert('L')
        
        if aleat:
            plt.figure(figsize=(10, 3))

            plt.subplot(1, 2, 1)
            plt.imshow(gray_img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(gray_img_copy, cmap='gray')
            plt.axis('off')

            plt.savefig("test3.png")

        pixels = gray_img.load()
        pixels_copy = gray_img_copy.load()
        img_mask = []
        label_mask = []
        
        MAX_PIXELS = max(MAX_PIXELS, RESIZED_WIDTH * RESIZED_WEIGHT)
        
        for x in range(RESIZED_WIDTH):
            for y in range(RESIZED_WEIGHT):
                img_mask.append(float(pixels[x, y] / 255))
                
                isInside = pixels_copy[x, y] != 0
                
                label_mask.append(1.0 if isInside else 0.0)
        
        data['mask'] = {
            'img': img_mask,
            'label': label_mask
        }
        
        if aleat:
            mask_img = Image.new("L", (RESIZED_WIDTH, RESIZED_WEIGHT))
            label_img = Image.new("L", (RESIZED_WIDTH, RESIZED_WEIGHT))
            
            i = 0
            for x in range(RESIZED_WIDTH):
                for y in range(RESIZED_WEIGHT):
                    mask_img.putpixel((x, y), int(255 * img_mask[i]))
                    label_img.putpixel((x, y), int(255 * label_mask[i]))
                    i += 1
            
            plt.figure(figsize=(10, 3))

            plt.subplot(1, 2, 1)
            plt.imshow(mask_img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(label_img, cmap='gray')
            plt.axis('off')

            plt.savefig("test4.png")
        
        aleat = False
        
        if cnt%100 == 0:
            print("{}/{} - {}".format(cnt//100, tot//100, MAX_PIXELS))
        cnt += 1
    
    return info

def showImage(bot, data, z):
    inp_img = Image.new("L", (RESIZED_WIDTH, RESIZED_WEIGHT))
    lbl_img = Image.new("L", (RESIZED_WIDTH, RESIZED_WEIGHT))
    out_img = Image.new("L", (RESIZED_WIDTH, RESIZED_WEIGHT))
    
    out_ans = bot.send(data['mask']['img'])
    
    num = "" if z >= 10 else "0"
    num += str(z)
    
    f = open("data/logs/value_{}.txt".format(num), "w")
    txt = ""

    i = 0
    for x in range(RESIZED_WIDTH):
        for y in range(RESIZED_WEIGHT):
            inp_img.putpixel((x, y), int(255 * data['mask']['img'][i]))
            lbl_img.putpixel((x, y), int(255 * data['mask']['label'][i]))
            out_img.putpixel((x, y), int(255 * out_ans[i]))
            txt += "{} - {}\n".format(data['mask']['label'][i], out_ans[i])
            i += 1
    
    f.write(txt)
    f.close()
    
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(inp_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(lbl_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(out_img, cmap='gray')
    plt.axis('off')

    plt.savefig("data/logs/result2_{}.png".format(num))

def main():
    dataset = buildDataset()
    train = maskImages(dataset[0:2000])
    test = maskImages(dataset[-100:])
    
    print("MAXPIXELS = {}".format(MAX_PIXELS))
    
    bot = Neural(
        sizes=[RESIZED_WEIGHT * RESIZED_WIDTH, 50, 50, RESIZED_WIDTH * RESIZED_WEIGHT],
        brain_path="data/facebrain.brain",
        eta=0.1,
        gpu=True,
        mini_batch_size=16,
        multilabel=True
    )
    
    bot.export("brain", "data/face")
    
    PERIOD = 20000
    
    for i in range(300):
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
            i,score/len(test), round(min_cost,4), round(avg_cost/len(test),4)
        ))
        
        j = 1
        for img in train:
            bot.learn(img['mask']['img'], img['mask']['label'])
            if j%PERIOD == 0:
                print("{}/{}".format(j//PERIOD, len(train)//PERIOD))
            j += 1
        if i % 10 == 0:
            showImage(bot, test[50], i//10)
    
    bot.export("brain", "data/face")

if __name__ == "__main__":
    main()
