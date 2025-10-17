/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#ifndef DAV1D_COMMON_H
#define DAV1D_COMMON_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#ifndef DAV1D_API
    #if defined _WIN32
      #if defined DAV1D_BUILDING_DLL
        #define DAV1D_API __declspec(dllexport)
      #else
        #define DAV1D_API
      #endif
    #else
      #if __GNUC__ >= 4
        #define DAV1D_API __attribute__ ((visibility ("default")))
      #else
        #define DAV1D_API
      #endif
    #endif
#endif

#if EPERM > 0
#define DAV1D_ERR(e) (-(e)) ///< Negate POSIX error code.
#else
#define DAV1D_ERR(e) (e)
#endif

/**
 * A reference-counted object wrapper for a user-configurable pointer.
 */
typedef struct Dav1dUserData {
    const uint8_t *data; ///< data pointer
    struct Dav1dRef *ref; ///< allocation origin
} Dav1dUserData;

/**
 * Input packet metadata which are copied from the input data used to
 * decode each image into the matching structure of the output image
 * returned back to the user. Since these are metadata fields, they
 * can be used for other purposes than the documented ones, they will
 * still be passed from input data to output picture without being
 * used internally.
 */
typedef struct Dav1dDataProps {
    int64_t timestamp; ///< container timestamp of input data, INT64_MIN if unknown (default)
    int64_t duration; ///< container duration of input data, 0 if unknown (default)
    int64_t offset; ///< stream offset of input data, -1 if unknown (default)
    size_t size; ///< packet size, default Dav1dData.sz
    struct Dav1dUserData user_data; ///< user-configurable data, default NULL members
} Dav1dDataProps;

/**
 * Release reference to a Dav1dDataProps.
 */
DAV1D_API void dav1d_data_props_unref(Dav1dDataProps *props);

#endif /* DAV1D_COMMON_H */
