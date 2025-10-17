/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
/* $Id: portlist.h,v 1.9 2007/06/19 23:47:17 tbox Exp $ */

/*! \file dns/portlist.h */

#include <isc/lang.h>
#include <isc/net.h>
#include <isc/types.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_portlist_create(isc_mem_t *mctx, dns_portlist_t **portlistp);
/*%<
 * Create a port list.
 * 
 * Requires:
 *\li	'mctx' to be valid.
 *\li	'portlistp' to be non NULL and '*portlistp' to be NULL;
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 *\li	#ISC_R_NOMEMORY
 *\li	#ISC_R_UNEXPECTED
 */

isc_result_t
dns_portlist_add(dns_portlist_t *portlist, int af, in_port_t port);
/*%<
 * Add the given <port,af> tuple to the portlist.
 *
 * Requires:
 *\li	'portlist' to be valid.
 *\li	'af' to be AF_INET or AF_INET6
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 *\li	#ISC_R_NOMEMORY
 */

void
dns_portlist_remove(dns_portlist_t *portlist, int af, in_port_t port);
/*%<
 * Remove the given <port,af> tuple to the portlist.
 *
 * Requires:
 *\li	'portlist' to be valid.
 *\li	'af' to be AF_INET or AF_INET6
 */

isc_boolean_t
dns_portlist_match(dns_portlist_t *portlist, int af, in_port_t port);
/*%<
 * Find the given <port,af> tuple to the portlist.
 *
 * Requires:
 *\li	'portlist' to be valid.
 *\li	'af' to be AF_INET or AF_INET6
 *
 * Returns
 * \li	#ISC_TRUE if the tuple is found, ISC_FALSE otherwise.
 */

void
dns_portlist_attach(dns_portlist_t *portlist, dns_portlist_t **portlistp);
/*%<
 * Attach to a port list.
 *
 * Requires:
 *\li	'portlist' to be valid.
 *\li	'portlistp' to be non NULL and '*portlistp' to be NULL;
 */

void
dns_portlist_detach(dns_portlist_t **portlistp);
/*%<
 * Detach from a port list.
 *
 * Requires:
 *\li	'*portlistp' to be valid.
 */

ISC_LANG_ENDDECLS
