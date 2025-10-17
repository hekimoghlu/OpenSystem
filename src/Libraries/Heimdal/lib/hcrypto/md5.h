/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

#ifndef HEIM_MD5_H
#define HEIM_MD5_H 1

/* symbol renaming */
#define MD5_Init hc_MD5_Init
#define MD5_Update hc_MD5_Update
#define MD5_Final hc_MD5_Final

/*
 *
 */

#define MD5_DIGEST_LENGTH 16

struct md5 {
  unsigned int sz[2];
  uint32_t counter[4];
  unsigned char save[64];
};

typedef struct md5 MD5_CTX;

void MD5_Init (struct md5 *m);
void MD5_Update (struct md5 *m, const void *p, size_t len);
void MD5_Final (void *res, struct md5 *m); /* uint32_t res[4] */

#endif /* HEIM_MD5_H */
