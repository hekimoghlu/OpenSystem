/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#ifndef PAS_FD_STREAM_H
#define PAS_FD_STREAM_H

#include "pas_stream.h"

PAS_BEGIN_EXTERN_C;

struct pas_fd_stream;
typedef struct pas_fd_stream pas_fd_stream;

struct pas_fd_stream {
    pas_stream base;
    int fd;
};

PAS_API extern pas_stream_functions pas_fd_stream_functions;

#define PAS_FD_STREAM_INITIALIZER(passed_fd) { \
        .base = PAS_STREAM_INITIALIZER(&pas_fd_stream_functions), \
        .fd = passed_fd \
    }

PAS_API extern pas_fd_stream pas_log_stream;

PAS_API void pas_fd_stream_construct(pas_fd_stream* stream, int fd);

PAS_API void pas_fd_stream_vprintf(pas_fd_stream* stream, const char* format, va_list arg_list) PAS_FORMAT_PRINTF(2, 0);

#define pas_fd_stream_printf(stream, ...) \
    pas_stream_printf((pas_stream*)(stream), __VA_ARGS__)

PAS_END_EXTERN_C;

#endif /* PAS_FD_STREAM_H */

