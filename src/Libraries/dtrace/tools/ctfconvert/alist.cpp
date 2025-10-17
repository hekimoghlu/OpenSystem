/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
 * Copyright 2001-2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Create, manage, and destroy association lists.  alists are arrays with
 * arbitrary index types, and are also commonly known as associative arrays.
 */

#include <stdio.h>
#include <stdlib.h>

#include "llvm-ADT/DenseMap.h"

#include "alist.h"
#include "memory.h"
#include "hash.h"
#include "ctftools.h"

struct alistDenseMapInfo {
	static inline void *getEmptyKey() {
		return reinterpret_cast<void *>(-1);
	}
	static inline void *getTombstoneKey() {
		return reinterpret_cast<void *>(-2);
	}
	static unsigned getHashValue(const void *v) {
		uintptr_t key = reinterpret_cast<uintptr_t>(v);
		return key * 37U;
	}
	static bool isEqual(const void *a, const void *b) {
		return a == b;
	}
};

extern "C" {

struct alist : public llvm::DenseMap<void *, void *, alistDenseMapInfo> {
	alist(unsigned size)
	: llvm::DenseMap<void *, void *, alistDenseMapInfo>(size)
	{
	}

	void
	stats(int verbose __unused)
	{
		printf("Alist statistics\n");
		printf(" Items  : %d\n", size());
	}
};

alist_t *
alist_new(unsigned size)
{
	return new alist{size};
}

void
alist_clear(alist_t *alist)
{
	alist->clear();
}

void
alist_free(alist_t *alist)
{
	delete alist;
}

void
alist_add(alist_t *alist, void *name, void *value)
{
	if (name == alistDenseMapInfo::getEmptyKey())
		terminate("Trying to insert the empty key (%p)", name);
	if (name == alistDenseMapInfo::getTombstoneKey())
		terminate("Trying to insert the tombstone key (%p)", name);
	alist->try_emplace(name, value);
}

int
alist_find(alist_t *alist, void *name, void **value)
{
	auto it = alist->find(name);

	if (it == alist->end())
		return (0);

	if (value)
		*value = it->second;

	return (1);
}

int
alist_iter(alist_t *alist, int (*func)(void *, void *, void *), void *priv)
{
	int cumrc = 0;
	int cbrc;

	for (auto it: *alist) {
		if ((cbrc = func(it.first, it.second, priv)) < 0)
			return (cbrc);
		cumrc += cbrc;
	}

	return (cumrc);
}

/*
 * Debugging support.  Used to print the contents of an alist.
 */

void
alist_stats(alist_t *alist, int verbose)
{
	printf("Alist statistics\n");
	alist->stats(verbose);
}

} // extern "C"
