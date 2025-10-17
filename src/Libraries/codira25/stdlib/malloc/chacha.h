/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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

#ifndef CHACHA_H
#define CHACHA_H

#include "util.h"

#define CHACHA_KEY_SIZE 32
#define CHACHA_IV_SIZE 8

typedef struct {
    u32 input[16];
} chacha_ctx;

void chacha_keysetup(chacha_ctx *x, const u8 *k);
void chacha_ivsetup(chacha_ctx *x, const u8 *iv);
void chacha_keystream_bytes(chacha_ctx *x, u8 *c, u32 bytes);

#endif
