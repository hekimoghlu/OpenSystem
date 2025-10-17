/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#ifndef CSET_H
#define	CSET_H

#include <stdbool.h>
#include <wchar.h>
#include <wctype.h>

struct csnode {
	wchar_t		csn_min;
	wchar_t		csn_max;
	struct csnode	*csn_left;
	struct csnode	*csn_right;
};

struct csclass {
	wctype_t	csc_type;
	bool		csc_invert;
	struct csclass	*csc_next;
};

struct cset {
#define	CS_CACHE_SIZE	256
	bool		cs_cache[CS_CACHE_SIZE];
	bool		cs_havecache;
	struct csclass	*cs_classes;
	struct csnode	*cs_root;
	bool		cs_invert;
};

bool			cset_addclass(struct cset *, wctype_t, bool);
struct cset *		cset_alloc(void);
bool 			cset_add(struct cset *, wchar_t);
void			cset_invert(struct cset *);
bool			cset_in_hard(struct cset *, wchar_t);
void			cset_cache(struct cset *);

static __inline bool
cset_in(struct cset *cs, wchar_t ch)
{

	if (ch < CS_CACHE_SIZE && cs->cs_havecache)
		return (cs->cs_cache[ch]);
	return (cset_in_hard(cs, ch));
}

#endif	/* CSET_H */
