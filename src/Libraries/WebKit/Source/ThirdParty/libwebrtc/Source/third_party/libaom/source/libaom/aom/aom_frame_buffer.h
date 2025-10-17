/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
#ifndef AOM_AOM_AOM_FRAME_BUFFER_H_
#define AOM_AOM_AOM_FRAME_BUFFER_H_

/*!\file
 * \brief Describes the decoder external frame buffer interface.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "aom/aom_integer.h"

/*!\brief The maximum number of work buffers used by libaom.
 *  Support maximum 4 threads to decode video in parallel.
 *  Each thread will use one work buffer.
 * TODO(hkuang): Add support to set number of worker threads dynamically.
 */
#define AOM_MAXIMUM_WORK_BUFFERS 8

/*!\brief The maximum number of reference buffers that a AV1 encoder may use.
 */
#define AOM_MAXIMUM_REF_BUFFERS 8

/*!\brief External frame buffer
 *
 * This structure holds allocated frame buffers used by the decoder.
 */
typedef struct aom_codec_frame_buffer {
  uint8_t *data; /**< Pointer to the data buffer */
  size_t size;   /**< Size of data in bytes */
  void *priv;    /**< Frame's private data */
} aom_codec_frame_buffer_t;

/*!\brief get frame buffer callback prototype
 *
 * This callback is invoked by the decoder to retrieve data for the frame
 * buffer in order for the decode call to complete. The callback must
 * allocate at least min_size in bytes and assign it to fb->data. The callback
 * must zero out all the data allocated. Then the callback must set fb->size
 * to the allocated size. The application does not need to align the allocated
 * data. The callback is triggered when the decoder needs a frame buffer to
 * decode a compressed image into. This function may be called more than once
 * for every call to aom_codec_decode. The application may set fb->priv to
 * some data which will be passed back in the aom_image_t and the release
 * function call. |fb| is guaranteed to not be NULL. On success the callback
 * must return 0. Any failure the callback must return a value less than 0.
 *
 * \param[in] priv         Callback's private data
 * \param[in] min_size     Size in bytes needed by the buffer
 * \param[in,out] fb       Pointer to aom_codec_frame_buffer_t
 */
typedef int (*aom_get_frame_buffer_cb_fn_t)(void *priv, size_t min_size,
                                            aom_codec_frame_buffer_t *fb);

/*!\brief release frame buffer callback prototype
 *
 * This callback is invoked by the decoder when the frame buffer is not
 * referenced by any other buffers. |fb| is guaranteed to not be NULL. On
 * success the callback must return 0. Any failure the callback must return
 * a value less than 0.
 *
 * \param[in] priv         Callback's private data
 * \param[in] fb           Pointer to aom_codec_frame_buffer_t
 */
typedef int (*aom_release_frame_buffer_cb_fn_t)(void *priv,
                                                aom_codec_frame_buffer_t *fb);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_AOM_FRAME_BUFFER_H_
