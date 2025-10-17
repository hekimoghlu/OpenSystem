/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include "compiler.h"


#include "nasmlib.h"
#include "error.h"

#if !defined(HAVE_VSNPRINTF) && !defined(HAVE__VSNPRINTF)

#define BUFFER_SIZE     65536   /* Bigger than any string we might print... */

static char snprintf_buffer[BUFFER_SIZE];

int vsnprintf(char *str, size_t size, const char *format, va_list ap)
{
    int rv, bytes;

    if (size > BUFFER_SIZE) {
        nasm_panic("vsnprintf: size (%llu) > BUFFER_SIZE (%d)",
                   (unsigned long long)size, BUFFER_SIZE);
        size = BUFFER_SIZE;
    }

    rv = vsprintf(snprintf_buffer, format, ap);
    if (rv >= BUFFER_SIZE)
        nasm_panic("vsnprintf buffer overflow");

    if (size > 0) {
        if ((size_t)rv < size-1)
            bytes = rv;
        else
            bytes = size-1;
        memcpy(str, snprintf_buffer, bytes);
        str[bytes] = '\0';
    }

    return rv;
}

#endif
