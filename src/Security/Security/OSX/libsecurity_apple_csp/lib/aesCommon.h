/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
//
// aesCommon.h - common AES/Rijndael constants
//
#ifndef _H_AES_COMMON_
#define _H_AES_COMMON_

#define MIN_AES_KEY_BITS		128
#define MID_AES_KEY_BITS		192
#define MAX_AES_KEY_BITS		256

#define MIN_AES_BLOCK_BITS		128
#define MID_AES_BLOCK_BITS		192
#define MAX_AES_BLOCK_BITS		256

#define MIN_AES_BLOCK_BYTES		(MIN_AES_BLOCK_BITS / 8)
#define DEFAULT_AES_BLOCK_BYTES	MIN_AES_BLOCK_BYTES

/*
 * When true, the Gladman AES implementation is present and is used
 * for all 128-bit block configurations.
 */
#define GLADMAN_AES_128_ENABLE	1

#endif	/* _H_AES_COMMON_ */
