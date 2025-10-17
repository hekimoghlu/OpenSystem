/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#ifndef _SEARCH_H_
#define _SEARCH_H_

#include <sys/cdefs.h>
#include <_types.h>
#include <sys/_types/_size_t.h>

typedef	struct entry {
	char	*key;
	void	*data;
} ENTRY;

typedef	enum {
	FIND, ENTER
} ACTION;

typedef	enum {
	preorder,
	postorder,
	endorder,
	leaf
} VISIT;

#ifdef _SEARCH_PRIVATE
typedef	struct node {
	char         *key;
	struct node  *llink, *rlink;
} node_t;

struct que_elem {
	struct que_elem *next;
	struct que_elem *prev;
};
#endif

__BEGIN_DECLS
int	 hcreate(size_t);
void	 hdestroy(void);
ENTRY	*hsearch(ENTRY, ACTION);
void	 insque(void *, void *);
void	*lfind(const void *, const void *, size_t *, size_t,
	    int (*)(const void *, const void *));
void	*lsearch(const void *, void *, size_t *, size_t,
	    int (*)(const void *, const void *));
void	 remque(void *);
void	*tdelete(const void * __restrict, void ** __restrict,
	    int (*)(const void *, const void *));
void	*tfind(const void *, void * const *,
	    int (*)(const void *, const void *));
void	*tsearch(const void *, void **, int (*)(const void *, const void *));
void	 twalk(const void *, void (*)(const void *, VISIT, int));
__END_DECLS

#endif /* !_SEARCH_H_ */
