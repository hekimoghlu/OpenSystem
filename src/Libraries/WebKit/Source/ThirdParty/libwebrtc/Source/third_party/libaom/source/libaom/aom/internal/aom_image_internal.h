/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
/*!\file
 * \brief Describes the internal functions associated with the aom image
 * descriptor.
 *
 */
#ifndef AOM_AOM_INTERNAL_AOM_IMAGE_INTERNAL_H_
#define AOM_AOM_INTERNAL_AOM_IMAGE_INTERNAL_H_

#include "aom/aom_image.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!\brief Array of aom_metadata structs for an image. */
struct aom_metadata_array {
  size_t sz;                       /* Number of metadata structs in the list */
  aom_metadata_t **metadata_array; /* Array of metadata structs */
};

/*!\brief Alloc memory for aom_metadata_array struct.
 *
 * Allocate memory for aom_metadata_array struct.
 * If sz is 0 the aom_metadata_array struct's internal buffer list will be
 * NULL, but the aom_metadata_array struct itself will still be allocated.
 * Returns a pointer to the allocated struct or NULL on failure.
 *
 * \param[in]    sz       Size of internal metadata list buffer
 */
aom_metadata_array_t *aom_img_metadata_array_alloc(size_t sz);

/*!\brief Free metadata array struct.
 *
 * Free metadata array struct and all metadata structs inside.
 *
 * \param[in]    arr       Metadata array struct pointer
 */
void aom_img_metadata_array_free(aom_metadata_array_t *arr);

typedef void *(*aom_alloc_img_data_cb_fn_t)(void *priv, size_t size);

/*!\brief Open a descriptor, allocating storage for the underlying image by
 * using the provided callback function.
 *
 * Returns a descriptor for storing an image of the given format. The storage
 * for the image is allocated by using the provided callback function. Unlike
 * aom_img_alloc(), the returned descriptor does not own the storage for the
 * image. The caller is responsible for freeing the storage for the image.
 *
 * Note: If the callback function is invoked and succeeds,
 * aom_img_alloc_with_cb() is guaranteed to succeed. Therefore, if
 * aom_img_alloc_with_cb() fails, the caller is assured that no storage was
 * allocated.
 *
 * \param[in]    img       Pointer to storage for descriptor. If this parameter
 *                         is NULL, the storage for the descriptor will be
 *                         allocated on the heap.
 * \param[in]    fmt       Format for the image
 * \param[in]    d_w       Width of the image
 * \param[in]    d_h       Height of the image
 * \param[in]    align     Alignment, in bytes, of the image buffer and
 *                         each row in the image (stride).
 * \param[in]    alloc_cb  Callback function used to allocate storage for the
 *                         image.
 * \param[in]    cb_priv   The first argument ('priv') for the callback
 *                         function.
 *
 * \return Returns a pointer to the initialized image descriptor. If the img
 *         parameter is non-null, the value of the img parameter will be
 *         returned.
 */
aom_image_t *aom_img_alloc_with_cb(aom_image_t *img, aom_img_fmt_t fmt,
                                   unsigned int d_w, unsigned int d_h,
                                   unsigned int align,
                                   aom_alloc_img_data_cb_fn_t alloc_cb,
                                   void *cb_priv);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_INTERNAL_AOM_IMAGE_INTERNAL_H_
