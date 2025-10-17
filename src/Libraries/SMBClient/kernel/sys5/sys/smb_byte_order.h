/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#ifndef _SMB_BYTEORDER_H_
#define _SMB_BYTEORDER_H_

#include <libkern/OSByteOrder.h>

#define htoles(x)	(OSSwapHostToLittleInt16(x))
#define letohs(x)	(OSSwapLittleToHostInt16(x))
#define	htolel(x)	(OSSwapHostToLittleInt32(x))
#define	letohl(x)	(OSSwapLittleToHostInt32(x))
#define	htoleq(x)	(OSSwapHostToLittleInt64(x))
#define	letohq(x)	(OSSwapLittleToHostInt64(x))

#define htobes(x)	(OSSwapHostToBigInt16(x))
#define betohs(x)	(OSSwapBigToHostInt16(x))
#define htobel(x)	(OSSwapHostToBigInt32(x))
#define betohl(x)	(OSSwapBigToHostInt32(x))
#define	htobeq(x)	(OSSwapHostToBigInt64(x))
#define	betohq(x)	(OSSwapBigToHostInt64(x))

#endif	/* !_SMB_BYTEORDER_H_ */
