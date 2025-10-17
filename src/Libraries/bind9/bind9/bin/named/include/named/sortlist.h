/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
/* $Id: sortlist.h,v 1.11 2007/06/19 23:46:59 tbox Exp $ */

#ifndef NAMED_SORTLIST_H
#define NAMED_SORTLIST_H 1

/*! \file */

#include <isc/types.h>

#include <dns/types.h>

/*%
 * Type for callback functions that rank addresses.
 */
typedef int 
(*dns_addressorderfunc_t)(const isc_netaddr_t *address, const void *arg);

/*%
 * Return value type for setup_sortlist.
 */
typedef enum {
	NS_SORTLISTTYPE_NONE,
	NS_SORTLISTTYPE_1ELEMENT,
	NS_SORTLISTTYPE_2ELEMENT
} ns_sortlisttype_t;

ns_sortlisttype_t
ns_sortlist_setup(dns_acl_t *acl, isc_netaddr_t *clientaddr,
		  const void **argp);
/*%<
 * Find the sortlist statement in 'acl' that applies to 'clientaddr', if any.
 *
 * If a 1-element sortlist item applies, return NS_SORTLISTTYPE_1ELEMENT and
 * make '*argp' point to the matching subelement.
 *
 * If a 2-element sortlist item applies, return NS_SORTLISTTYPE_2ELEMENT and
 * make '*argp' point to ACL that forms the second element.
 *
 * If no sortlist item applies, return NS_SORTLISTTYPE_NONE and set '*argp'
 * to NULL.
 */

int
ns_sortlist_addrorder1(const isc_netaddr_t *addr, const void *arg);
/*%<
 * Find the sort order of 'addr' in 'arg', the matching element
 * of a 1-element top-level sortlist statement.
 */

int
ns_sortlist_addrorder2(const isc_netaddr_t *addr, const void *arg);
/*%<
 * Find the sort order of 'addr' in 'arg', a topology-like
 * ACL forming the second element in a 2-element top-level
 * sortlist statement.
 */

void
ns_sortlist_byaddrsetup(dns_acl_t *sortlist_acl, isc_netaddr_t *client_addr,
			dns_addressorderfunc_t *orderp,
			const void **argp);
/*%<
 * Find the sortlist statement in 'acl' that applies to 'clientaddr', if any.
 * If a sortlist statement applies, return in '*orderp' a pointer to a function
 * for ranking network addresses based on that sortlist statement, and in
 * '*argp' an argument to pass to said function.  If no sortlist statement
 * applies, set '*orderp' and '*argp' to NULL.
 */

#endif /* NAMED_SORTLIST_H */
