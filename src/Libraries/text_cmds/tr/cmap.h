/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#ifndef CMAP_H
#define	CMAP_H

#include <limits.h>
#include <stdbool.h>
#include <wchar.h>

struct cmapnode {
	wint_t		cmn_from;
	wint_t		cmn_to;
	struct cmapnode	*cmn_left;
	struct cmapnode	*cmn_right;
};

struct cmap {
#define	CM_CACHE_SIZE	128
	wint_t		cm_cache[CM_CACHE_SIZE];
	bool		cm_havecache;
	struct cmapnode	*cm_root;
#define	CM_DEF_SELF	-2
	wint_t		cm_def;
	wint_t		cm_min;
	wint_t		cm_max;
};

struct cmap *	cmap_alloc(void);
bool		cmap_add(struct cmap *, wint_t, wint_t);
wint_t		cmap_lookup_hard(struct cmap *, wint_t);
void		cmap_cache(struct cmap *);
wint_t		cmap_default(struct cmap *, wint_t);

static __inline wint_t
cmap_lookup(struct cmap *cm, wint_t from)
{

	if (from < CM_CACHE_SIZE && cm->cm_havecache)
		return (cm->cm_cache[from]);
	return (cmap_lookup_hard(cm, from));
}

static __inline wint_t
cmap_min(struct cmap *cm)
{

	return (cm->cm_min);
}

static __inline wint_t
cmap_max(struct cmap *cm)
{

	return (cm->cm_max);
}

#endif
