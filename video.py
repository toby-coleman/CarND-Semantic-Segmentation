import tensorflow as tf
from moviepy.editor import VideoFileClip
import scipy.misc
import os.path
import numpy as np


def convert_frame(sess, logits, keep_prob, image_pl, image_shape, img_in):
    """
    Annotate a single frame
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param image_shape: Tuple - Shape of image
    :param img_in: Input image
    :return: Annotated image
    """
    image = scipy.misc.imresize(img_in, image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


def convert_video(sess, logits, keep_prob, image_pl, image_shape, video_file):
    clip = VideoFileClip(video_file)
    out_file = os.path.splitext(video_file)[0] + '_processed.mp4'

    clip_processed = clip.fl_image(lambda x: convert_frame(sess, logits,
                                                           keep_prob, image_pl, image_shape, x))
    print('Writing to ' + out_file)
    clip_processed.write_videofile(out_file, audio=False)
