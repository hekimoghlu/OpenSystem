/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#include "krb5_locl.h"

static krb5_error_code
rr13(unsigned char *buf, size_t len)
{
    unsigned char *tmp;
    size_t bytes = (len + 7) / 8;
    size_t i;
    if(len == 0)
	return 0;
    {
	const int bits = 13 % len;
	const int lbit = len % 8;

	tmp = malloc(bytes);
	if (tmp == NULL)
	    return ENOMEM;
	memcpy(tmp, buf, bytes);
	if(lbit) {
	    /* pad final byte with inital bits */
	    tmp[bytes - 1] &= 0xff << (8 - lbit);
	    for(i = lbit; i < 8; i += len)
		tmp[bytes - 1] |= buf[0] >> i;
	}
	for(i = 0; i < bytes; i++) {
	    ssize_t bb;
	    ssize_t b1, s1, b2, s2;
	    /* calculate first bit position of this byte */
	    bb = 8 * i - bits;
	    while(bb < 0)
		bb += len;
	    /* byte offset and shift count */
	    b1 = bb / 8;
	    s1 = bb % 8;

	    if((size_t)bb + 8 > bytes * 8)
		/* watch for wraparound */
		s2 = (len + 8 - s1) % 8;
	    else
		s2 = 8 - s1;
	    b2 = (b1 + 1) % bytes;
	    buf[i] = (tmp[b1] << s1) | (tmp[b2] >> s2);
	}
	free(tmp);
    }
    return 0;
}

/* Add `b' to `a', both being one's complement numbers. */
static void
add1(unsigned char *a, unsigned char *b, size_t len)
{
    ssize_t i;
    int carry = 0;
    for(i = len - 1; i >= 0; i--){
	int x = a[i] + b[i] + carry;
	carry = x > 0xff;
	a[i] = x & 0xff;
    }
    for(i = len - 1; carry && i >= 0; i--){
	int x = a[i] + carry;
	carry = x > 0xff;
	a[i] = x & 0xff;
    }
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
_krb5_n_fold(const void *str, size_t len, void *key, size_t size)
{
    /* if len < size we need at most N * len bytes, ie < 2 * size;
       if len > size we need at most 2 * len */
    krb5_error_code ret = 0;
    size_t maxlen = 2 * max(size, len);
    size_t l = 0;
    unsigned char *tmp = malloc(maxlen);
    unsigned char *buf = malloc(len);

    if (tmp == NULL || buf == NULL) {
        ret = ENOMEM;
	goto out;
    }

    memcpy(buf, str, len);
    memset(key, 0, size);
    do {
	memcpy(tmp + l, buf, len);
	l += len;
	ret = rr13(buf, len * 8);
	if (ret)
	    goto out;
	while(l >= size) {
	    add1(key, tmp, size);
	    l -= size;
	    if(l == 0)
		break;
	    memmove(tmp, tmp + size, l);
	}
    } while(l != 0);
out:
    if (buf) {
        memset(buf, 0, len);
	free(buf);
    }
    if (tmp) {
        memset(tmp, 0, maxlen);
	free(tmp);
    }
    return ret;
}
