import os
import glob
import random
from shutil import  copy, move

mods = ['vision', 'touch']

for folder in glob.glob('/data/yao/apps/hw/ObjectFolder*-*/'):
    for obj in os.listdir(folder):
        for d in range(1,5):
            cls = f'{obj}-{d}'
            
            copy(f'{folder}{obj}/vision/00{d-1}.png',
                    f'./data/vision/{cls}.png')
            
            copy(f'{folder}{obj}/touch/{d}.png',
                    f'./data/touch/{cls}.png')
            

## split data  8:2
vfiles = random.sample(glob.glob('data/vision/*.png'), 4000)
tfiles = [f.replace('vision','touch') for f in vfiles]

for file in vfiles[:3200]:
    ff = file.replace('vision', 'vision/train')
    move(file, ff)

for file in tfiles[:3200]:
    ff = file.replace('touch', 'touch/train')
    move(file, ff)

for file in vfiles[3200:]:
    ff = file.replace('vision', 'vision/val')
    move(file, ff)

for file in tfiles[3200:]:
    ff = file.replace('touch', 'touch/val')
    move(file, ff)


