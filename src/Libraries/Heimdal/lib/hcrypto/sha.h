/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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

#ifndef HEIM_SHA_H
#define HEIM_SHA_H 1

/* symbol renaming */
#define SHA1_Init hc_SHA1_Init
#define SHA1_Update hc_SHA1_Update
#define SHA1_Final hc_SHA1_Final
#define SHA256_Init hc_SHA256_Init
#define SHA256_Update hc_SHA256_Update
#define SHA256_Final hc_SHA256_Final
#define SHA384_Init hc_SHA384_Init
#define SHA384_Update hc_SHA384_Update
#define SHA384_Final hc_SHA384_Final
#define SHA512_Init hc_SHA512_Init
#define SHA512_Update hc_SHA512_Update
#define SHA512_Final hc_SHA512_Final

/*
 * SHA-1
 */

#define SHA_DIGEST_LENGTH 20

struct sha {
  unsigned int sz[2];
  uint32_t counter[5];
  unsigned char save[64];
};

typedef struct sha SHA_CTX;

void SHA1_Init (struct sha *m);
void SHA1_Update (struct sha *m, const void *v, size_t len);
void SHA1_Final (void *res, struct sha *m);

/*
 * SHA-2 256
 */

#define SHA256_DIGEST_LENGTH 32

struct hc_sha256state {
  unsigned int sz[2];
  uint32_t counter[8];
  unsigned char save[64];
};

typedef struct hc_sha256state SHA256_CTX;

void SHA256_Init (SHA256_CTX *);
void SHA256_Update (SHA256_CTX *, const void *, size_t);
void SHA256_Final (void *, SHA256_CTX *);

/*
 * SHA-2 512
 */

#define SHA512_DIGEST_LENGTH 64

struct hc_sha512state {
  uint64_t sz[2];
  uint64_t counter[8];
  unsigned char save[128];
};

typedef struct hc_sha512state SHA512_CTX;

void SHA512_Init (SHA512_CTX *);
void SHA512_Update (SHA512_CTX *, const void *, size_t);
void SHA512_Final (void *, SHA512_CTX *);

#define SHA384_DIGEST_LENGTH 48

typedef struct hc_sha512state SHA384_CTX;

void SHA384_Init (SHA384_CTX *);
void SHA384_Update (SHA384_CTX *, const void *, size_t);
void SHA384_Final (void *, SHA384_CTX *);

#endif /* HEIM_SHA_H */
