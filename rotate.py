from PIL import Image
#for img_name in ['pack2/IMG_00{}.JPG'.format(i) for i in range(18, 29)]:
for i in range(18, 29):
    name = 'pack2/IMG_00{}.JPG'.format(i)
    new_name = 'pack2_rot/IMG_00{}.JPG'.format(i)
    im = Image.open(name)
    im = im.rotate(270, expand=True)
    im.save(new_name)
