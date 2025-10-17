/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#ifndef _SHA_H_
#define _SHA_H_		1

#include <CommonCrypto/CommonDigest.h>

#define	SHA_CBLOCK	CC_SHA1_BLOCK_BYTES
#define	SHA_LBLOCK	16
#define	SHA_BLOCK	16
#define	SHA_LAST_BLOCK  56
#define	SHA_LENGTH_BLOCK 8
#define	SHA_DIGEST_LENGTH CC_SHA1_DIGEST_LENGTH

#define SHA_CTX		CC_SHA1_CTX
#define SHA1_CTX	CC_SHA1_CTX

#include <sys/cdefs.h>
#include <sys/types.h>

#define SHA1_Init	CC_SHA1_Init
#define SHA1_Update	CC_SHA1_Update
#define SHA1_Final	CC_SHA1_Final

__BEGIN_DECLS
int     SHA_Init(SHA_CTX *);
int     SHA_Update(SHA_CTX *, const void *, size_t);
int     SHA_Final(unsigned char *md, SHA_CTX *);

char   *SHA_End(SHA_CTX *, char *);
char   *SHA_Fd(int, char *);
char   *SHA_FdChunk(int, char *, off_t, off_t);
char   *SHA_File(const char *, char *);
char   *SHA_FileChunk(const char *, char *, off_t, off_t);
char   *SHA_Data(const void *, unsigned int, char *);

char   *SHA1_End(SHA_CTX *, char *);
char   *SHA1_Fd(int, char *);
char   *SHA1_FdChunk(int, char *, off_t, off_t);
char   *SHA1_File(const char *, char *);
char   *SHA1_FileChunk(const char *, char *, off_t, off_t);
char   *SHA1_Data(const void *, unsigned int, char *);
__END_DECLS

#endif /* !_SHA_H_ */
