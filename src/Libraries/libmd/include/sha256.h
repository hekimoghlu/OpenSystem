/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#ifndef _SHA256_H_
#define _SHA256_H_

#include <CommonCrypto/CommonDigest.h>

#define SHA256_BLOCK_LENGTH		CC_SHA256_BLOCK_BYTES
#define SHA256_DIGEST_LENGTH		CC_SHA256_DIGEST_LENGTH
#define SHA256_DIGEST_STRING_LENGTH	(SHA256_DIGEST_LENGTH * 2 + 1)

#define	SHA256_CTX	CC_SHA256_CTX

#include <sys/cdefs.h>
#include <sys/types.h>

#define SHA256_Init	CC_SHA256_Init
#define SHA256_Update	CC_SHA256_Update
#define SHA256_Final	CC_SHA256_Final

__BEGIN_DECLS
char   *SHA256_End(SHA256_CTX *, char *);
char   *SHA256_Data(const void *, unsigned int, char *);
char   *SHA256_Fd(int, char *);
char   *SHA256_FdChunk(int, char *, off_t, off_t);
char   *SHA256_File(const char *, char *);
char   *SHA256_FileChunk(const char *, char *, off_t, off_t);
__END_DECLS

#endif /* !_SHA256_H_ */
