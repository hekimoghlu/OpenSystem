/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#ifndef _CORECRYPTO_CCANSIKDF_H_
#define _CORECRYPTO_CCANSIKDF_H_

#include <corecrypto/ccdigest.h>
#include <corecrypto/cc_priv.h>

CC_NONNULL((1, 3, 7))
int ccansikdf_x963(const struct ccdigest_info *di,
                   const size_t Z_len, const unsigned char *Z,
                   const size_t sharedinfo_byte_len,
		   const void *sharedinfo, const size_t key_len,
		   uint8_t *key);

#endif
