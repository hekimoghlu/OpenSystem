/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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

#ifndef _FR_SHA1_H
#define _FR_SHA1_H

#include <stdint.h>
typedef struct {
    uint32_t state[5];
    uint32_t count[2];
    uint8_t buffer[64];
} fr_SHA1_CTX;

void fr_SHA1Transform(fr_SHA1_CTX * context, const uint8_t buffer[64]);
void fr_SHA1Init(fr_SHA1_CTX* context);
#if 0
void fr_SHA1Update(fr_SHA1_CTX* context, const uint8_t* data, unsigned int len);
void fr_SHA1Final(uint8_t digest[20], fr_SHA1_CTX* context);
#endif /* 0 */

/*
 * this version implements a raw SHA1 transform, no length is appended,
 * nor any 128s out to the block size.
 */
void fr_SHA1FinalNoLen(uint8_t digest[20], fr_SHA1_CTX* context);

#endif /* _FR_SHA1_H */
