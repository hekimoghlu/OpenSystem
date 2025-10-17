/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
/*
 * Glenn Fowler
 * AT&T Research
 *
 * hash table library
 */

#include "hashlib.h"

/*
 * apply walker to each active bucket in the table
 */

int
hashwalk(Hash_table_t* tab, int flags, register int (*walker)(const char*, char*, void*), void* handle)
{
	register Hash_bucket_t*	b;
	register int		v;
	Hash_position_t*	pos;

	if (!(pos = hashscan(tab, flags)))
		return(-1);
	v = 0;
	while (b = hashnext(pos))
		if ((v = (*walker)(hashname(b), (tab->flags & HASH_VALUE) ? b->value : (char*)b, handle)) < 0)
			break;
	hashdone(pos);
	return(v);
}
