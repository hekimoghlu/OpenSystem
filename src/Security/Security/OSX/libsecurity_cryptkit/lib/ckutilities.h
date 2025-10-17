/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#ifndef	_CK_UTILITIES_H_
#define _CK_UTILITIES_H_

#include "giantIntegers.h"
#include "elliptic.h"
#include "feeTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned char *mem_from_giant(giant x, unsigned *memLen);
giant giant_with_data(const unsigned char *d, int len);

void serializeGiant(giant g,
	unsigned char *cp,
	unsigned numBytes);

void deserializeGiant(const unsigned char *cp,
	giant g,
	unsigned numBytes);

#ifdef __cplusplus
}
#endif

#endif	/* _CK_UTILITIES_H_ */
