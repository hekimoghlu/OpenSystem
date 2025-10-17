/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
#include <config.h>
#include "roken.h"
#include <ctype.h>
#include "hex.h"

static const char hexchar_upper[16] = "0123456789ABCDEF";
static const char hexchar_lower[16] = "0123456789abcdef";

static int
pos(char c)
{
    const char *p;
    c = toupper((unsigned char)c);
    for (p = hexchar_upper; *p; p++)
	if (*p == c)
	    return (int)(p - hexchar_upper);
    return -1;
}

static ssize_t ROKEN_LIB_CALL
encode_hex(const void *data, size_t size, const char *hexchar, char **str)
{
    const unsigned char *q = data;
    size_t i;
    char *p;

    /* check for overflow */
    if (size * 2 < size) {
        *str = NULL;
	return -1;
    }

    p = malloc(size * 2 + 1);
    if (p == NULL) {
        *str = NULL;
	return -1;
    }

    for (i = 0; i < size; i++) {
	p[i * 2] = hexchar[(*q >> 4) & 0xf];
	p[i * 2 + 1] = hexchar[*q & 0xf];
	q++;
    }
    p[i * 2] = '\0';
    *str = p;

    return i * 2;
}


ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
hex_encode(const void *data, size_t size, char **str)
{
    return encode_hex(data, size, hexchar_upper, str);
}

ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
hex_encode_lower(const void *data, size_t size, char **str)
{
    return encode_hex(data, size, hexchar_lower, str);
}

ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
hex_decode(const char *str, void *data, size_t len)
{
    size_t l;
    unsigned char *p = data;
    size_t i;

    l = strlen(str);

    /* check for overflow, same as (l+1)/2 but overflow safe */
    if ((l/2) + (l&1) > len)
	return -1;

    if (l & 1) {
	p[0] = pos(str[0]);
	str++;
	p++;
    }
    for (i = 0; i < l / 2; i++)
	p[i] = pos(str[i * 2]) << 4 | pos(str[(i * 2) + 1]);
    return i + (l & 1);
}
