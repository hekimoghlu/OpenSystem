/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef DAV1D_SRC_PICTURE_H
#define DAV1D_SRC_PICTURE_H

#include <stdatomic.h>

#include "src/thread.h"
#include "dav1d/picture.h"

#include "src/thread_data.h"
#include "src/ref.h"

enum PlaneType {
    PLANE_TYPE_Y,
    PLANE_TYPE_UV,
    PLANE_TYPE_BLOCK,
    PLANE_TYPE_ALL,
};

enum PictureFlags {
    PICTURE_FLAG_NEW_SEQUENCE =       1 << 0,
    PICTURE_FLAG_NEW_OP_PARAMS_INFO = 1 << 1,
    PICTURE_FLAG_NEW_TEMPORAL_UNIT  = 1 << 2,
};

typedef struct Dav1dThreadPicture {
    Dav1dPicture p;
    int visible;
    enum PictureFlags flags;
    // [0] block data (including segmentation map and motion vectors)
    // [1] pixel data
    atomic_uint *progress;
} Dav1dThreadPicture;

typedef struct Dav1dPictureBuffer {
    void *data;
    struct Dav1dPictureBuffer *next;
} Dav1dPictureBuffer;

/*
 * Allocate a picture with custom border size.
 */
int dav1d_thread_picture_alloc(Dav1dContext *c, Dav1dFrameContext *f, const int bpc);

/**
 * Allocate a picture with identical metadata to an existing picture.
 * The width is a separate argument so this function can be used for
 * super-res, where the width changes, but everything else is the same.
 * For the more typical use case of allocating a new image of the same
 * dimensions, use src->p.w as width.
 */
int dav1d_picture_alloc_copy(Dav1dContext *c, Dav1dPicture *dst, const int w,
                             const Dav1dPicture *src);

/**
 * Create a copy of a picture.
 */
void dav1d_picture_ref(Dav1dPicture *dst, const Dav1dPicture *src);
void dav1d_thread_picture_ref(Dav1dThreadPicture *dst,
                              const Dav1dThreadPicture *src);
void dav1d_thread_picture_move_ref(Dav1dThreadPicture *dst,
                                   Dav1dThreadPicture *src);
void dav1d_thread_picture_unref(Dav1dThreadPicture *p);

/**
 * Move a picture reference.
 */
void dav1d_picture_move_ref(Dav1dPicture *dst, Dav1dPicture *src);

int dav1d_default_picture_alloc(Dav1dPicture *p, void *cookie);
void dav1d_default_picture_release(Dav1dPicture *p, void *cookie);
void dav1d_picture_unref_internal(Dav1dPicture *p);

/**
 * Get event flags from picture flags.
 */
enum Dav1dEventFlags dav1d_picture_get_event_flags(const Dav1dThreadPicture *p);

#endif /* DAV1D_SRC_PICTURE_H */
