/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#ifndef PAS_STREAM_H
#define PAS_STREAM_H

#include "pas_utils.h"
#include <stdarg.h>

PAS_BEGIN_EXTERN_C;

/* pas_stream is a low-level string print stream that uses printf-like APIs like FILE. */

struct pas_stream;
struct pas_stream_functions;
typedef struct pas_stream pas_stream;
typedef struct pas_stream_functions pas_stream_functions;

struct pas_stream {
    const pas_stream_functions* functions;
};

struct pas_stream_functions {
    void (*vprintf)(pas_stream* stream, const char* format, va_list) PAS_FORMAT_PRINTF(2, 0);
};

#define PAS_STREAM_INITIALIZER(passed_functions) { .functions = (passed_functions) }

PAS_API void pas_stream_vprintf(pas_stream* stream, const char* format, va_list) PAS_FORMAT_PRINTF(2, 0);
PAS_API void pas_stream_printf(pas_stream* stream, const char* format, ...) PAS_FORMAT_PRINTF(2, 3);

static inline void pas_stream_print_comma(pas_stream* stream, bool* comma, const char* string)
{
    if (!*comma)
        *comma = true;
    else
        pas_stream_printf(stream, "%s", string);
}

PAS_END_EXTERN_C;

#endif /* PAS_STREAM_H */

