/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#ifndef THIRD_PARTY_ZLIB_GOOGLE_COMPRESSION_UTILS_PORTABLE_H_
#define THIRD_PARTY_ZLIB_GOOGLE_COMPRESSION_UTILS_PORTABLE_H_

#include <stdint.h>

/* TODO(cavalcantii): remove support for Chromium ever building with a system
 * zlib.
 */
#if defined(USE_SYSTEM_ZLIB)
#    include <zlib.h>
/* AOSP build requires relative paths. */
#else
#    include "zlib.h"
#endif

namespace zlib_internal
{

enum WrapperType
{
    ZLIB,
    GZIP,
    ZRAW,
};

uLongf GzipExpectedCompressedSize(uLongf input_size);

uint32_t GetGzipUncompressedSize(const Bytef *compressed_data, size_t length);

int GzipCompressHelper(Bytef *dest,
                       uLongf *dest_length,
                       const Bytef *source,
                       uLong source_length,
                       void *(*malloc_fn)(size_t),
                       void (*free_fn)(void *));

int CompressHelper(WrapperType wrapper_type,
                   Bytef *dest,
                   uLongf *dest_length,
                   const Bytef *source,
                   uLong source_length,
                   int compression_level,
                   void *(*malloc_fn)(size_t),
                   void (*free_fn)(void *));

int GzipUncompressHelper(Bytef *dest,
                         uLongf *dest_length,
                         const Bytef *source,
                         uLong source_length);

int UncompressHelper(WrapperType wrapper_type,
                     Bytef *dest,
                     uLongf *dest_length,
                     const Bytef *source,
                     uLong source_length);

}  // namespace zlib_internal

#endif  // THIRD_PARTY_ZLIB_GOOGLE_COMPRESSION_UTILS_PORTABLE_H_
