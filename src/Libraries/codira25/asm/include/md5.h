/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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

#ifndef MD5_H
#define MD5_H

#include "compiler.h"

#define MD5_HASHBYTES 16

typedef struct MD5Context {
	uint32_t buf[4];
	uint32_t bits[2];
	unsigned char in[64];
} MD5_CTX;

extern void   MD5Init(MD5_CTX *context);
extern void   MD5Update(MD5_CTX *context, unsigned char const *buf,
	       unsigned len);
extern void   MD5Final(unsigned char digest[MD5_HASHBYTES], MD5_CTX *context);
extern void   MD5Transform(uint32_t buf[4], uint32_t const in[16]);
extern char * MD5End(MD5_CTX *, char *);

#endif /* !MD5_H */
