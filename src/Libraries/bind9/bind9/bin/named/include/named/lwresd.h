/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
/* $Id: lwresd.h,v 1.19 2007/06/19 23:46:59 tbox Exp $ */

#ifndef NAMED_LWRESD_H
#define NAMED_LWRESD_H 1

/*! \file */

#include <isc/types.h>
#include <isc/sockaddr.h>

#include <isccfg/cfg.h>

#include <dns/types.h>

struct ns_lwresd {
	unsigned int magic;

	isc_mutex_t lock;
	dns_view_t *view;
	ns_lwsearchlist_t *search;
	unsigned int ndots;
	isc_mem_t *mctx;
	isc_boolean_t shutting_down;
	unsigned int refs;
};

struct ns_lwreslistener {
	unsigned int magic;

	isc_mutex_t lock;
	isc_mem_t *mctx;
	isc_sockaddr_t address;
	ns_lwresd_t *manager;
	isc_socket_t *sock;
	unsigned int refs;
	ISC_LIST(ns_lwdclientmgr_t) cmgrs;
	ISC_LINK(ns_lwreslistener_t) link;
};

/*%
 * Configure lwresd.
 */
isc_result_t
ns_lwresd_configure(isc_mem_t *mctx, const cfg_obj_t *config);

isc_result_t
ns_lwresd_parseeresolvconf(isc_mem_t *mctx, cfg_parser_t *pctx,
			   cfg_obj_t **configp);

/*%
 * Trigger shutdown.
 */
void
ns_lwresd_shutdown(void);

/*
 * Manager functions
 */
/*% create manager */
isc_result_t
ns_lwdmanager_create(isc_mem_t *mctx, const cfg_obj_t *lwres,
		      ns_lwresd_t **lwresdp);

/*% attach to manager */
void
ns_lwdmanager_attach(ns_lwresd_t *source, ns_lwresd_t **targetp);

/*% detach from manager */
void
ns_lwdmanager_detach(ns_lwresd_t **lwresdp);

/*
 * Listener functions
 */
/*% attach to listener */
void
ns_lwreslistener_attach(ns_lwreslistener_t *source,
			ns_lwreslistener_t **targetp);

/*% detach from lister */
void
ns_lwreslistener_detach(ns_lwreslistener_t **listenerp);

/*% link client manager */
void
ns_lwreslistener_unlinkcm(ns_lwreslistener_t *listener, ns_lwdclientmgr_t *cm);

/*% unlink client manager */
void
ns_lwreslistener_linkcm(ns_lwreslistener_t *listener, ns_lwdclientmgr_t *cm);




/*
 * INTERNAL FUNCTIONS.
 */
void *
ns__lwresd_memalloc(void *arg, size_t size);

void
ns__lwresd_memfree(void *arg, void *mem, size_t size);

#endif /* NAMED_LWRESD_H */
