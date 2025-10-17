/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
/*
 * This private header exports 3 APIs.
 *	_OSRosettaCheck() -     An inline function that returns true if we are
 *				currently running under Rosetta.
 *	IF_ROSETTA() -		Which is used to as a regular conditional
 *				expression that is true only if the current
 *				code is executing in the Rosetta
 *				translation space.
 *	ROSETTA_ONLY(exprs) -   Which is used to create a block code that only
 *				executes if we are running in Rosetta.
 *
 * for example
 *
 * IF_ROSETTA() {
 *	// Do Cross endian swapping of input data
 *	outdata = OSSwap??(indata);
 * }
 * else {
 *      // Do straight through
 *	outdata = indata;
 * }
 *
 * outdata = indata;
 * ROSETTA_ONLY(
 *	// Do Cross endian swapping of input data
 *	outdata = OSSwap??(outdata);
 * );
 */

#ifndef _LIBKERN_OSCROSSENDIAN_H
#define _LIBKERN_OSCROSSENDIAN_H

#include <sys/sysctl.h>

static __inline__ int
_OSRosettaCheck(void)
{
	return 0;
}

#define IF_ROSETTA() if (__builtin_expect(_OSRosettaCheck(), 0) )

#define ROSETTA_ONLY(exprs)     \
do {                            \
    IF_ROSETTA() {              \
	exprs                   \
    }                           \
} while(0)

#endif /*  _LIBKERN_OSCROSSENDIAN_H */
