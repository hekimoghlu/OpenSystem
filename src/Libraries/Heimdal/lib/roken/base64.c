/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "base64.h"

static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static int
pos(char c)
{
    const char *p;
    for (p = base64_chars; *p; p++)
	if (*p == c)
	    return (int)(p - base64_chars);
    return -1;
}

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
base64_encode(const void *data, int size, char **str)
{
    char *s, *p;
    int i;
    int c;
    const unsigned char *q;

    if (size > INT_MAX/4 || size < 0) {
	*str = NULL;
	return -1;
    }

    p = s = (char *) malloc(size * 4 / 3 + 4);
    if (p == NULL) {
        *str = NULL;
	return -1;
    }
    q = (const unsigned char *) data;

    for (i = 0; i < size;) {
	c = q[i++];
	c *= 256;
	if (i < size)
	    c += q[i];
	i++;
	c *= 256;
	if (i < size)
	    c += q[i];
	i++;
	p[0] = base64_chars[(c & 0x00fc0000) >> 18];
	p[1] = base64_chars[(c & 0x0003f000) >> 12];
	p[2] = base64_chars[(c & 0x00000fc0) >> 6];
	p[3] = base64_chars[(c & 0x0000003f) >> 0];
	if (i > size)
	    p[3] = '=';
	if (i > size + 1)
	    p[2] = '=';
	p += 4;
    }
    *p = 0;
    *str = s;
    return (int) strlen(s);
}

#define DECODE_ERROR 0xffffffff

static unsigned int
token_decode(const char *token)
{
    int i;
    unsigned int val = 0;
    int marker = 0;
    if (strlen(token) < 4)
	return DECODE_ERROR;
    for (i = 0; i < 4; i++) {
	val *= 64;
	if (token[i] == '=')
	    marker++;
	else if (marker > 0)
	    return DECODE_ERROR;
	else
	    val += pos(token[i]);
    }
    if (marker > 2)
	return DECODE_ERROR;
    return (marker << 24) | val;
}

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
base64_decode(const char *str, void *data)
{
    const char *p;
    unsigned char *q;

    q = data;
    for (p = str; *p && (*p == '=' || strchr(base64_chars, *p)); p += 4) {
	unsigned int val = token_decode(p);
	unsigned int marker = (val >> 24) & 0xff;
	if (val == DECODE_ERROR)
	    return -1;
	*q++ = (val >> 16) & 0xff;
	if (marker < 2)
	    *q++ = (val >> 8) & 0xff;
	if (marker < 1)
	    *q++ = val & 0xff;
    }
    return (int)(q - (unsigned char *) data);
}
