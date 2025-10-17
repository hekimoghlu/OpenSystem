/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#ifndef _MM_HASH_H
#define _MM_HASH_H
#include "mm.h"

typedef void (*MM_HashDtor)(void *);

typedef struct _MM_Bucket {
	struct _MM_Bucket *next;
	char *key;
	int length;
	unsigned int hash;
	void *data;
} MM_Bucket;

#define MM_HASH_SIZE 1009
typedef struct _Hash {
	 MM_Bucket *buckets[ MM_HASH_SIZE ];
	 MM *mm;
	 MM_HashDtor dtor;
	 int nElements;
} MM_Hash;

MM_Hash *mm_hash_new(MM *, MM_HashDtor);
void mm_hash_free(MM_Hash *table);
void *mm_hash_find(MM_Hash *table, const void *key, int length);
void mm_hash_add(MM_Hash *table, char *key, int length, void *data);
void mm_hash_delete(MM_Hash *table, char *key, int length);
void mm_hash_update(MM_Hash *table, char *key, int length, void *data);
#endif

/*
Local variables:
tab-width: 4
c-basic-offset: 4
End:
vim600: noet sw=4 ts=4 fdm=marker
vim<600: noet sw=4 ts=4
*/
