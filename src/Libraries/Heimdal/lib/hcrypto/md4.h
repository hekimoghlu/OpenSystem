/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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

#ifndef HEIM_MD4_H
#define HEIM_MD4_H 1

/* symbol renaming */
#define MD4_Init hc_MD4_Init
#define MD4_Update hc_MD4_Update
#define MD4_Final hc_MD4_Final

/*
 *
 */

#define MD4_DIGEST_LENGTH 16

struct md4 {
  unsigned int sz[2];
  uint32_t counter[4];
  unsigned char save[64];
};

typedef struct md4 MD4_CTX;

void MD4_Init (struct md4 *m);
void MD4_Update (struct md4 *m, const void *p, size_t len);
void MD4_Final (void *res, struct md4 *m);

#endif /* HEIM_MD4_H */
