/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#ifndef	_AES_BOXES_H_
#define _AES_BOXES_H_

#include "rijndael-alg-ref.h"

#ifdef	__cplusplus
extern "C" {
#endif

#define AES_MUL_BY_LOOKUP	1

#if			AES_MUL_BY_LOOKUP
extern const word8 mulBy0x02[256];
extern const word8 mulBy0x03[256];
extern const word8 mulBy0x0e[256];
extern const word8 mulBy0x0b[256];
extern const word8 mulBy0x0d[256];
extern const word8 mulBy0x09[256];
#else
extern const unsigned char Logtable[256];
extern const unsigned char Alogtable[256];
#endif	/* AES_MUL_BY_LOOKUP */

extern const unsigned char S[256];
extern const unsigned char Si[256];
extern const unsigned char iG[4][4];
extern const unsigned long rcon[30];

#ifdef	__cplusplus
}
#endif

#endif	/* _AES_BOXES_H_ */
