from PIL import Image
import sys

im = Image.open(sys.argv[1])
rgb_im = im.convert('RGB')
pix = rgb_im.load()
for i in range(im.size[0]):
	for j in range(im.size[1]):
		r, g, b = rgb_im.getpixel((i, j))
		r, g, b = int(r/2), int(g/2), int(b/2)
		pix[i, j] = r, g, b

rgb_im.save("Q2.png")
