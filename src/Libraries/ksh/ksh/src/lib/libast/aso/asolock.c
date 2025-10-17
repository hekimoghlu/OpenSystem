/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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
#pragma prototyped

#include "asohdr.h"

#if defined(_UWIN) && defined(_BLD_ast)

NoN(asolock)

#else

int
asolock(unsigned int volatile* lock, unsigned int key, int type)
{
	unsigned int	k;

	if (key)
		switch (type)
		{
		case ASO_UNLOCK:
			return *lock == 0 ? 0 : asocasint(lock, key, 0) == key ? 0 : -1;
		case ASO_TRYLOCK:
			return *lock == key ? 0 : asocasint(lock, 0, key) == 0 ? 0 : -1;
		case ASO_LOCK:
			if (*lock == key)
				return 0;
			/*FALLTHROUGH*/
		case ASO_SPINLOCK:
			for (k = 0; asocasint(lock, 0, key) != 0; ASOLOOP(k));
			return 0;
		}
	return -1;
}

#endif
