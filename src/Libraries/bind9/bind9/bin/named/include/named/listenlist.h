/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
/* $Id: listenlist.h,v 1.15 2007/06/19 23:46:59 tbox Exp $ */

#ifndef NAMED_LISTENLIST_H
#define NAMED_LISTENLIST_H 1

/*****
 ***** Module Info
 *****/

/*! \file
 * \brief
 * "Listen lists", as in the "listen-on" configuration statement.
 */

/***
 *** Imports
 ***/
#include <isc/net.h>

#include <dns/types.h>

/***
 *** Types
 ***/

typedef struct ns_listenelt ns_listenelt_t;
typedef struct ns_listenlist ns_listenlist_t;

struct ns_listenelt {
	isc_mem_t *	       		mctx;
	in_port_t			port;
	isc_dscp_t			dscp;  /* -1 = not set, 0..63 */
	dns_acl_t *	       		acl;
	ISC_LINK(ns_listenelt_t)	link;
};

struct ns_listenlist {
	isc_mem_t *			mctx;
	int				refcount;
	ISC_LIST(ns_listenelt_t)	elts;
};

/***
 *** Functions
 ***/

isc_result_t
ns_listenelt_create(isc_mem_t *mctx, in_port_t port, isc_dscp_t dscp,
		    dns_acl_t *acl, ns_listenelt_t **target);
/*%
 * Create a listen-on list element.
 */

void
ns_listenelt_destroy(ns_listenelt_t *elt);
/*%
 * Destroy a listen-on list element.
 */

isc_result_t
ns_listenlist_create(isc_mem_t *mctx, ns_listenlist_t **target);
/*%
 * Create a new, empty listen-on list.
 */

void
ns_listenlist_attach(ns_listenlist_t *source, ns_listenlist_t **target);
/*%
 * Attach '*target' to '*source'.
 */

void
ns_listenlist_detach(ns_listenlist_t **listp);
/*%
 * Detach 'listp'.
 */

isc_result_t
ns_listenlist_default(isc_mem_t *mctx, in_port_t port, isc_dscp_t dscp,
		      isc_boolean_t enabled, ns_listenlist_t **target);
/*%
 * Create a listen-on list with default contents, matching
 * all addresses with port 'port' (if 'enabled' is ISC_TRUE),
 * or no addresses (if 'enabled' is ISC_FALSE).
 */

#endif /* NAMED_LISTENLIST_H */


