/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#ifndef YASM_MD5_H
#define YASM_MD5_H

#ifndef YASM_LIB_DECL
#define YASM_LIB_DECL
#endif

/* Unlike previous versions of this code, uint32 need not be exactly
   32 bits, merely 32 bits or more.  Choosing a data type which is 32
   bits instead of 64 is not important; speed is considerably more
   important.  ANSI guarantees that "unsigned long" will be big enough,
   and always using it seems to have few disadvantages.  */

typedef struct yasm_md5_context {
        unsigned long buf[4];
        unsigned long bits[2];
        unsigned char in[64];
} yasm_md5_context;

YASM_LIB_DECL
void yasm_md5_init(yasm_md5_context *context);
YASM_LIB_DECL
void yasm_md5_update(yasm_md5_context *context, unsigned char const *buf,
                     unsigned long len);
YASM_LIB_DECL
void yasm_md5_final(unsigned char digest[16], yasm_md5_context *context);
YASM_LIB_DECL
void yasm_md5_transform(unsigned long buf[4], const unsigned char in[64]);

#endif /* !YASM_MD5_H */
