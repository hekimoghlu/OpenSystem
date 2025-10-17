/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include <sys/endian.h>
#include "_fpmath.h"

union IEEEf2bits {
	float	f;
	struct {
#if _BYTE_ORDER == _LITTLE_ENDIAN
		unsigned int	man	:23;
		unsigned int	exp	:8;
		unsigned int	sign	:1;
#else /* _BIG_ENDIAN */
		unsigned int	sign	:1;
		unsigned int	exp	:8;
		unsigned int	man	:23;
#endif
	} bits;
};

#define	DBL_MANH_SIZE	20
#define	DBL_MANL_SIZE	32

union IEEEd2bits {
	double	d;
	struct {
#if _BYTE_ORDER == _LITTLE_ENDIAN
		unsigned int	manl	:32;
		unsigned int	manh	:20;
		unsigned int	exp	:11;
		unsigned int	sign	:1;
#else /* _BIG_ENDIAN */
		unsigned int	sign	:1;
		unsigned int	exp	:11;
		unsigned int	manh	:20;
		unsigned int	manl	:32;
#endif
	} bits;
};
