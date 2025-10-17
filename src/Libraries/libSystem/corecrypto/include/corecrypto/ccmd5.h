/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
#ifndef _CORECRYPTO_CCMD5_H_
#define _CORECRYPTO_CCMD5_H_

#include <corecrypto/ccdigest.h>

#define CCMD5_BLOCK_SIZE   64
#define CCMD5_OUTPUT_SIZE  16
#define CCMD5_STATE_SIZE   16

extern const uint32_t ccmd5_initial_state[4];

/* Selector */
const struct ccdigest_info *ccmd5_di(void);

/* Implementations */
extern const struct ccdigest_info ccmd5_ltc_di;

#endif /* _CORECRYPTO_CCMD5_H_ */
