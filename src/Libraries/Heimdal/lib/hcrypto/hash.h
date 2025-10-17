/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
/* $Id$ */

/* stuff in common between md4, md5, and sha1 */

#ifndef __hash_h__
#define __hash_h__

#ifdef KRB5
#include <krb5-types.h>
#endif
#include <roken.h>

#ifndef min
#define min(a,b) (((a)>(b))?(b):(a))
#endif

/* Vector Crays doesn't have a good 32-bit type, or more precisely,
   int32_t as defined by <bind/bitypes.h> isn't 32 bits, and we don't
   want to depend in being able to redefine this type.  To cope with
   this we have to clamp the result in some places to [0,2^32); no
   need to do this on other machines.  Did I say this was a mess?
   */

#ifdef _CRAY
#define CRAYFIX(X) ((X) & 0xffffffff)
#else
#define CRAYFIX(X) (X)
#endif

static inline uint32_t
cshift (uint32_t x, unsigned int n)
{
    x = CRAYFIX(x);
    return CRAYFIX((x << n) | (x >> (32 - n)));
}

static inline uint64_t
cshift64 (uint64_t x, unsigned int n)
{
  return ((uint64_t)x << (uint64_t)n) | ((uint64_t)x >> ((uint64_t)64 - (uint64_t)n));
}

#endif /* __hash_h__ */
