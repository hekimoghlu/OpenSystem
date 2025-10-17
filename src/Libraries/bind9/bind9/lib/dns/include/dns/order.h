/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
/* $Id: order.h,v 1.9 2007/06/19 23:47:17 tbox Exp $ */

#ifndef DNS_ORDER_H
#define DNS_ORDER_H 1

/*! \file dns/order.h */

#include <isc/lang.h>
#include <isc/types.h>

#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
dns_order_create(isc_mem_t *mctx, dns_order_t **orderp);
/*%<
 * Create a order object.
 *
 * Requires:
 * \li	'orderp' to be non NULL and '*orderp == NULL'.
 *\li	'mctx' to be valid.
 *
 * Returns:
 *\li	ISC_R_SUCCESS
 *\li	ISC_R_NOMEMORY
 */

isc_result_t
dns_order_add(dns_order_t *order, dns_name_t *name,
	      dns_rdatatype_t rdtype, dns_rdataclass_t rdclass,
	      unsigned int mode);
/*%<
 * Add a entry to the end of the order list.
 *
 * Requires:
 * \li	'order' to be valid.
 *\li	'name' to be valid.
 *\li	'mode' to be one of #DNS_RDATASERATTR_RANDOMIZE,
 *		#DNS_RDATASERATTR_RANDOMIZE or zero (#DNS_RDATASERATTR_CYCLIC).
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 *\li	#ISC_R_NOMEMORY
 */

unsigned int
dns_order_find(dns_order_t *order, dns_name_t *name,
	       dns_rdatatype_t rdtype, dns_rdataclass_t rdclass);
/*%<
 * Find the first matching entry on the list.
 *
 * Requires:
 *\li	'order' to be valid.
 *\li	'name' to be valid.
 *
 * Returns the mode set by dns_order_add() or zero.
 */

void
dns_order_attach(dns_order_t *source, dns_order_t **target);
/*%<
 * Attach to the 'source' object.
 *
 * Requires:
 * \li	'source' to be valid.
 *\li	'target' to be non NULL and '*target == NULL'.
 */

void
dns_order_detach(dns_order_t **orderp);
/*%<
 * Detach from the object.  Clean up if last this was the last
 * reference.
 *
 * Requires:
 *\li	'*orderp' to be valid.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_ORDER_H */
