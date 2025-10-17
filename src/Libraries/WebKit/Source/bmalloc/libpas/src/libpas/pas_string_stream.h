/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifndef PAS_STRING_STREAM_H
#define PAS_STRING_STREAM_H

#include "pas_allocation_config.h"
#include "pas_stream.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* pas_stream is a low-level string print stream that uses printf-like APIs like FILE. */

struct pas_string_stream;
typedef struct pas_string_stream pas_string_stream;

struct pas_string_stream {
    pas_stream base;
    pas_allocation_config allocation_config;
    char* buffer;
    size_t next;
    size_t size;
    char inline_buffer[128];
};

PAS_API void pas_string_stream_construct(pas_string_stream* stream,
                                         const pas_allocation_config* allocation_config);
PAS_API void pas_string_stream_destruct(pas_string_stream* stream);
PAS_API void pas_string_stream_reset(pas_string_stream* stream);

PAS_API void pas_string_stream_vprintf(pas_string_stream* stream, const char* format, va_list) PAS_FORMAT_PRINTF(2, 0);

#define pas_string_stream_printf(stream, ...) \
    pas_stream_printf((pas_stream*)(stream), __VA_ARGS__)

/* Returns a string that is live so long as you don't destruct the stream. */
static inline const char* pas_string_stream_get_string(pas_string_stream* stream)
{
    return stream->buffer;
}

static inline size_t pas_string_stream_get_string_length(pas_string_stream* stream)
{
    return stream->next;
}

PAS_END_EXTERN_C;

#endif /* PAS_STRING_STREAM_H */

