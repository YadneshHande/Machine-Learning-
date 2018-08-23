from scipy import misc
import tensorflow as tf

img = misc.imread('AMFED/AMFED/happiness/0d48e11a-2f87-4626-9c30-46a2e54ce58e.flv_7.0_5.jpg')
print img.shape    # (32, 32, 3)

img_tf = tf.Variable(img)
print img_tf.get_shape().as_list()  # [32, 32, 3]


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
im = sess.run(img_tf)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(im)
fig.add_subplot(1,2,2)
plt.imshow(img)
plt.show()