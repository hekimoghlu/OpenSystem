/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#ifndef _SHA512_H_
#define _SHA512_H_

#include <CommonCrypto/CommonDigest.h>

#define SHA512_BLOCK_LENGTH		CC_SHA512_BLOCK_BYTES
#define SHA512_DIGEST_LENGTH		CC_SHA512_DIGEST_LENGTH
#define SHA512_DIGEST_STRING_LENGTH	(SHA512_DIGEST_LENGTH * 2 + 1)

#define SHA512_CTX	CC_SHA512_CTX

#include <sys/cdefs.h>
#include <sys/types.h>

#define SHA512_Init	CC_SHA512_Init
#define SHA512_Update	CC_SHA512_Update
#define SHA512_Final	CC_SHA512_Final

__BEGIN_DECLS
char   *SHA512_End(SHA512_CTX *, char *);
char   *SHA512_Data(const void *, unsigned int, char *);
char   *SHA512_Fd(int, char *);
char   *SHA512_FdChunk(int, char *, off_t, off_t);
char   *SHA512_File(const char *, char *);
char   *SHA512_FileChunk(const char *, char *, off_t, off_t);
__END_DECLS

#endif /* !_SHA512_H_ */
