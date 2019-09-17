import struct
import imghdr
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib
from multiprocessing import Pool
import pandas as pd
import numpy as np

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            raise AssertionError('imghead len != 24')
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                raise AssertionError('png check failed')
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                raise
        else:
            print(fname, imghdr.what(fname))
            #raise AssertionError('file format not supported')
            img = cv2.imread(fname)
            print(img.shape)
            height, width, _ = img.shape

        return width, height

def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError("encode_binary_mask expects a binary mask, received dtype == %s" % mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError("encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape)
    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode('utf-8')


def parallel_apply(df, func, n_cores=24):
    #ncores = 24
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def general_ensemble(dets, iou_thresh = 0.5, weights=None):
    assert(type(iou_thresh) == float)
    
    ndets = len(dets)
    
    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)
        
        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()
    
    for idet in range(0,ndets):
        det = dets[idet]
        for box in det:
            if box[2] in used:
                continue
                
            used.append(box[2])
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]
                
                if iodet == idet:
                    continue
                
                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox[2] in used:
                        # Not already used
                        if box[1] == obox[1]:
                            # Same class
                            iou = computeIOU(box[0], obox[0])
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox
                                
                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox,w))
                    used.append(bestbox[2])
                            
            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                #new_box[2] /= ndets
                new_box[2] *= weights[idet]
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)
                
                conf = 0.0
                
                #wsum = 0.0
                masks, mask_w = [], []
                for bb in allboxes:
                    w = bb[1]
                    #wsum += w
                    b = bb[0]
                    conf += w*b[2]
                    masks.append(b[0].astype(np.float32))
                    mask_w.append(w)
                
                #new_mask = (np.mean(masks, 0) > 0.51).astype(np.uint8)
                new_mask = (np.average(masks, 0, weights=mask_w) > 0.5).astype(np.uint8)

                new_box = [new_mask, box[1], conf]
                out.append(new_box)
    return out


def computeIOU(mask1, mask2):
    
    intersect_area = ((mask1 * mask2) > 0).sum()
    
    iou = intersect_area / ((mask1 + mask2) > 0).sum()
    return iou
