/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "memory.h"
#include "atom.h"
#include "llvm-ADT/DenseSet.h"

llvm::DenseSet<const char *> atoms{4096};

extern "C" {

atom_t *
atom_get(const char *s)
{
	auto it = atoms.insert(s);
	if (it.second) {
		*it.first = xstrdup(s);
	}
	return reinterpret_cast<atom_t *>(*it.first);
}

atom_t *
atom_get_consume(char *s)
{
	auto it = atoms.insert(s);
	if (!it.second) {
		free(s);
	}
	return reinterpret_cast<atom_t *>(*it.first);
}

__attribute__((always_inline)) // let LTO know
unsigned
atom_hash(atom_t *atom)
{
	unsigned long key = reinterpret_cast<unsigned long>(atom);
	key ^= key >> 4;
#if __LP64__
	key *= 0x8a970be7488fda55;
	key ^= __builtin_bswap64(key);
#else
	key *= 0x5052acdb;
	key ^= __builtin_bswap32(key);
#endif
	return static_cast<unsigned>(key);
}

} // extern "C"
