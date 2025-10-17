/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
 * change table size and rehash
 * size must be a power of 2
 */

void
hashsize(register Hash_table_t* tab, int size)
{
	register Hash_bucket_t**	old_s;
	register Hash_bucket_t**	new_s;
	register Hash_bucket_t*		old_b;
	register Hash_bucket_t*		new_b;
	Hash_bucket_t**			old_sx;
	unsigned int			index;
	Hash_region_f			region;
	void*				handle;

	if (size > 0 && size != tab->size && !(size & (size - 1)))
	{
		if (region = tab->root->local->region)
		{
			handle = tab->root->local->handle;
			new_s = (Hash_bucket_t**)(*region)(handle, NiL, sizeof(Hash_bucket_t*) * size, 0);
		}
		else new_s = newof(0, Hash_bucket_t*, size, 0);
		if (!new_s) tab->flags |= HASH_FIXED;
		else
		{
			old_sx = (old_s = tab->table) + tab->size;
			tab->size = size;
			while (old_s < old_sx)
			{
				old_b = *old_s++;
				while (old_b)
				{
					new_b = old_b;
					old_b = old_b->next;
					index = new_b->hash;
					HASHMOD(tab, index);
					new_b->next = new_s[index];
					new_s[index] = new_b;
				}
			}
			if ((tab->flags & (HASH_RESIZE|HASH_STATIC)) != HASH_STATIC)
			{
				if (region) (*region)(handle, tab->table, 0, 0);
				else free(tab->table);
			}
			tab->table = new_s;
			tab->flags |= HASH_RESIZE;
		}
	}
}
