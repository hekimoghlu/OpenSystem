/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
/* implemented from description in draft-kaukonen-cipher-arcfour-03.txt */

#include "config.h"

#include <rc4.h>

#define SWAP(k,x,y)				\
{ unsigned int _t; 				\
  _t = k->state[x]; 				\
  k->state[x] = k->state[y]; 			\
  k->state[y] = _t;				\
}

void
RC4_set_key(RC4_KEY *key, const int len, const unsigned char *data)
{
    int i, j;

    for (i = 0; i < 256; i++)
	key->state[i] = i;
    for (i = 0, j = 0; i < 256; i++) {
	j = (j + key->state[i] + data[i % len]) % 256;
	SWAP(key, i, j);
    }
    key->x = key->y = 0;
}

void
RC4(RC4_KEY *key, const int len, const unsigned char *in, unsigned char *out)
{
    int i, t;
    unsigned x, y;

    x = key->x;
    y = key->y;
    for (i = 0; i < len; i++) {
	x = (x + 1) % 256;
	y = (y + key->state[x]) % 256;
	SWAP(key, x, y);
	t = (key->state[x] + key->state[y]) % 256;
	*out++ = key->state[t] ^ *in++;
    }
    key->x = x;
    key->y = y;
}
