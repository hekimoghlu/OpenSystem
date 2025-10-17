/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
/* $Id: sortlist.c,v 1.17 2007/09/14 01:46:05 marka Exp $ */

/*! \file */

#include <config.h>

#include <isc/mem.h>
#include <isc/util.h>

#include <dns/acl.h>
#include <dns/result.h>

#include <named/globals.h>
#include <named/server.h>
#include <named/sortlist.h>

ns_sortlisttype_t
ns_sortlist_setup(dns_acl_t *acl, isc_netaddr_t *clientaddr,
		  const void **argp)
{
	unsigned int i;

	if (acl == NULL)
		goto dont_sort;

	for (i = 0; i < acl->length; i++) {
		/*
		 * 'e' refers to the current 'top level statement'
		 * in the sortlist (see ARM).
		 */
		dns_aclelement_t *e = &acl->elements[i];
		dns_aclelement_t *try_elt;
		dns_aclelement_t *order_elt = NULL;
		const dns_aclelement_t *matched_elt = NULL;

		if (e->type == dns_aclelementtype_nestedacl) {
			dns_acl_t *inner = e->nestedacl;

			if (inner->length == 0)
				try_elt = e;
			else if (inner->length > 2)
				goto dont_sort;
			else if (inner->elements[0].negative)
				goto dont_sort;
			else {
				try_elt = &inner->elements[0];
				if (inner->length == 2)
					order_elt = &inner->elements[1];
			}
		} else {
			/*
			 * BIND 8 allows bare elements at the top level
			 * as an undocumented feature.
			 */
			try_elt = e;
		}

		if (dns_aclelement_match(clientaddr, NULL, try_elt,
					 &ns_g_server->aclenv,
					 &matched_elt)) {
			if (order_elt != NULL) {
				if (order_elt->type ==
				    dns_aclelementtype_nestedacl) {
					*argp = order_elt->nestedacl;
					return (NS_SORTLISTTYPE_2ELEMENT);
				} else if (order_elt->type ==
					   dns_aclelementtype_localhost &&
					   ns_g_server->aclenv.localhost != NULL) {
					*argp = ns_g_server->aclenv.localhost;
					return (NS_SORTLISTTYPE_2ELEMENT);
				} else if (order_elt->type ==
					   dns_aclelementtype_localnets &&
					   ns_g_server->aclenv.localnets != NULL) {
					*argp = ns_g_server->aclenv.localnets;
					return (NS_SORTLISTTYPE_2ELEMENT);
				} else {
					/*
					 * BIND 8 allows a bare IP prefix as
					 * the 2nd element of a 2-element
					 * sortlist statement.
					 */
					*argp = order_elt;
					return (NS_SORTLISTTYPE_1ELEMENT);
				}
			} else {
				INSIST(matched_elt != NULL);
				*argp = matched_elt;
				return (NS_SORTLISTTYPE_1ELEMENT);
			}
		}
	}

	/* No match; don't sort. */
 dont_sort:
	*argp = NULL;
	return (NS_SORTLISTTYPE_NONE);
}

int
ns_sortlist_addrorder2(const isc_netaddr_t *addr, const void *arg) {
	const dns_acl_t *sortacl = (const dns_acl_t *) arg;
	int match;

	(void)dns_acl_match(addr, NULL, sortacl,
			    &ns_g_server->aclenv,
			    &match, NULL);
	if (match > 0)
		return (match);
	else if (match < 0)
		return (INT_MAX - (-match));
	else
		return (INT_MAX / 2);
}

int
ns_sortlist_addrorder1(const isc_netaddr_t *addr, const void *arg) {
	const dns_aclelement_t *matchelt = (const dns_aclelement_t *) arg;
	if (dns_aclelement_match(addr, NULL, matchelt,
				 &ns_g_server->aclenv,
				 NULL)) {
		return (0);
	} else {
		return (INT_MAX);
	}
}

void
ns_sortlist_byaddrsetup(dns_acl_t *sortlist_acl, isc_netaddr_t *client_addr,
		       dns_addressorderfunc_t *orderp,
		       const void **argp)
{
	ns_sortlisttype_t sortlisttype;

	sortlisttype = ns_sortlist_setup(sortlist_acl, client_addr, argp);

	switch (sortlisttype) {
	case NS_SORTLISTTYPE_1ELEMENT:
		*orderp = ns_sortlist_addrorder1;
		break;
	case NS_SORTLISTTYPE_2ELEMENT:
		*orderp = ns_sortlist_addrorder2;
		break;
	case NS_SORTLISTTYPE_NONE:
		*orderp = NULL;
		break;
	default:
		UNEXPECTED_ERROR(__FILE__, __LINE__,
				 "unexpected return from ns_sortlist_setup(): "
				 "%d", sortlisttype);
		break;
	}
}

