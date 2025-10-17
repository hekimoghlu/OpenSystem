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
/* bit twiddling keygen for p12 password based encryption */

#include <CoreFoundation/CFString.h>

typedef enum {
    PBE_ID_Key      = 1,
    PBE_ID_IV       = 2,
    PBE_ID_MAC      = 3
} P12_PBE_ID;

int p12_pbe_gen(CFStringRef passphrase, uint8_t *salt_ptr, size_t salt_length, 
    unsigned iter_count, P12_PBE_ID pbe_id, uint8_t *data, size_t length,
    unsigned int hash_blocksize, unsigned int hash_outputsize);
