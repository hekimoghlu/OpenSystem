/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef _SHA256_H
#define _SHA256_H

#if HAVE_INTTYPES_H
# include <inttypes.h>
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#define SHA256_HASH_SIZE 32

/* Hash size in 32-bit words */
#define SHA256_HASH_WORDS 8

#ifdef _MSC_VER
typedef unsigned __int64    uint64_t;
#elif !(defined(__hpux) && defined(__ia64))
typedef unsigned long long  uint64_t;
#endif

#if !(defined(__hpux) && defined(__ia64))
typedef unsigned int        uint32_t;
typedef unsigned char       uint8_t;
#endif

struct _SHA256Context {
  uint64_t totalLength;
  uint32_t hash[SHA256_HASH_WORDS];
  uint32_t bufferLength;
  union {
    uint32_t words[16];
    uint8_t bytes[64];
  } buffer;
};

typedef struct _SHA256Context SHA256Context;
typedef struct _SHA256Context SHA256_CTX;

#ifdef __cplusplus
extern "C" {
#endif

void SHA256Init (SHA256Context *sc);
void SHA256Update (SHA256Context *sc, const void *data, uint32_t len);
void SHA256Final (SHA256Context *sc, uint8_t hash[SHA256_HASH_SIZE]);

void SHA224Init (SHA256Context *sc);

#ifdef __cplusplus
}
#endif

#endif /* !_SHA256_H */
