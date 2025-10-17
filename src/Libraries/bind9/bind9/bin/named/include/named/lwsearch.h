/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
/* $Id: lwsearch.h,v 1.9 2007/06/19 23:46:59 tbox Exp $ */

#ifndef NAMED_LWSEARCH_H
#define NAMED_LWSEARCH_H 1

#include <isc/mutex.h>
#include <isc/result.h>
#include <isc/types.h>

#include <dns/types.h>

#include <named/types.h>

/*! \file
 * \brief
 * Lightweight resolver search list types and routines.
 *
 * An ns_lwsearchlist_t holds a list of search path elements.
 *
 * An ns_lwsearchctx stores the state of search list during a lookup
 * operation.
 */

/*% An ns_lwsearchlist_t holds a list of search path elements. */
struct ns_lwsearchlist {
	unsigned int magic;

	isc_mutex_t lock;
	isc_mem_t *mctx;
	unsigned int refs;
	dns_namelist_t names;
};
/*% An ns_lwsearchctx stores the state of search list during a lookup operation. */
struct ns_lwsearchctx {
	dns_name_t *relname;
	dns_name_t *searchname;
	unsigned int ndots;
	ns_lwsearchlist_t *list;
	isc_boolean_t doneexact;
	isc_boolean_t exactfirst;
};

isc_result_t
ns_lwsearchlist_create(isc_mem_t *mctx, ns_lwsearchlist_t **listp);
/*%<
 * Create an empty search list object.
 */

void
ns_lwsearchlist_attach(ns_lwsearchlist_t *source, ns_lwsearchlist_t **target);
/*%<
 * Attach to a search list object.
 */

void
ns_lwsearchlist_detach(ns_lwsearchlist_t **listp);
/*%<
 * Detach from a search list object.
 */

isc_result_t
ns_lwsearchlist_append(ns_lwsearchlist_t *list, dns_name_t *name);
/*%<
 * Append an element to a search list.  This creates a copy of the name.
 */

void
ns_lwsearchctx_init(ns_lwsearchctx_t *sctx, ns_lwsearchlist_t *list,
		    dns_name_t *name, unsigned int ndots);
/*%<
 * Creates a search list context structure.
 */

void
ns_lwsearchctx_first(ns_lwsearchctx_t *sctx);
/*%<
 * Moves the search list context iterator to the first element, which
 * is usually the exact name.
 */

isc_result_t
ns_lwsearchctx_next(ns_lwsearchctx_t *sctx);
/*%<
 * Moves the search list context iterator to the next element.
 */

isc_result_t
ns_lwsearchctx_current(ns_lwsearchctx_t *sctx, dns_name_t *absname);
/*%<
 * Obtains the current name to be looked up.  This involves either
 * concatenating the name with a search path element, making an
 * exact name absolute, or doing nothing.
 */

#endif /* NAMED_LWSEARCH_H */
