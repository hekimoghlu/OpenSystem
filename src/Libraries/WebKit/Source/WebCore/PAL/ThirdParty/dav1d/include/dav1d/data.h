/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#ifndef DAV1D_DATA_H
#define DAV1D_DATA_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

typedef struct Dav1dData {
    const uint8_t *data; ///< data pointer
    size_t sz; ///< data size
    struct Dav1dRef *ref; ///< allocation origin
    Dav1dDataProps m; ///< user provided metadata passed to the output picture
} Dav1dData;

/**
 * Allocate data.
 *
 * @param data Input context.
 * @param   sz Size of the data that should be allocated.
 *
 * @return Pointer to the allocated buffer on success. NULL on error.
 */
DAV1D_API uint8_t * dav1d_data_create(Dav1dData *data, size_t sz);

/**
 * Wrap an existing data array.
 *
 * @param          data Input context.
 * @param           buf The data to be wrapped.
 * @param            sz Size of the data.
 * @param free_callback Function to be called when we release our last
 *                      reference to this data. In this callback, $buf will be
 *                      the $buf argument to this function, and $cookie will
 *                      be the $cookie input argument to this function.
 * @param        cookie Opaque parameter passed to free_callback().
 *
 * @return 0 on success. A negative DAV1D_ERR value on error.
 */
DAV1D_API int dav1d_data_wrap(Dav1dData *data, const uint8_t *buf, size_t sz,
                              void (*free_callback)(const uint8_t *buf, void *cookie),
                              void *cookie);

/**
 * Wrap a user-provided data pointer into a reference counted object.
 *
 * data->m.user_data field will initialized to wrap the provided $user_data
 * pointer.
 *
 * $free_callback will be called on the same thread that released the last
 * reference. If frame threading is used, make sure $free_callback is
 * thread-safe.
 *
 * @param          data Input context.
 * @param     user_data The user data to be wrapped.
 * @param free_callback Function to be called when we release our last
 *                      reference to this data. In this callback, $user_data
 *                      will be the $user_data argument to this function, and
 *                      $cookie will be the $cookie input argument to this
 *                      function.
 * @param        cookie Opaque parameter passed to $free_callback.
 *
 * @return 0 on success. A negative DAV1D_ERR value on error.
 */
DAV1D_API int dav1d_data_wrap_user_data(Dav1dData *data,
                                        const uint8_t *user_data,
                                        void (*free_callback)(const uint8_t *user_data,
                                                              void *cookie),
                                        void *cookie);

/**
 * Free the data reference.
 *
 * The reference count for data->m.user_data will be decremented (if it has been
 * initialized with dav1d_data_wrap_user_data). The $data object will be memset
 * to 0.
 *
 * @param data Input context.
 */
DAV1D_API void dav1d_data_unref(Dav1dData *data);

#endif /* DAV1D_DATA_H */
