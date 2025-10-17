/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
/*! \file */

#if defined(LIBC_SCCS) && !defined(lint)
static char rcsid[] =
	"$Id: netscope.c,v 1.13 2007/06/19 23:47:17 tbox Exp $";
#endif /* LIBC_SCCS and not lint */

#include <config.h>

#include <isc/string.h>
#include <isc/net.h>
#include <isc/netscope.h>
#include <isc/result.h>

isc_result_t
isc_netscope_pton(int af, char *scopename, void *addr, isc_uint32_t *zoneid) {
	char *ep;
#ifdef ISC_PLATFORM_HAVEIFNAMETOINDEX
	unsigned int ifid;
#endif
	struct in6_addr *in6;
	isc_uint32_t zone;
	isc_uint64_t llz;

	/* at this moment, we only support AF_INET6 */
	if (af != AF_INET6)
		return (ISC_R_FAILURE);

	in6 = (struct in6_addr *)addr;

	/*
	 * Basically, "names" are more stable than numeric IDs in terms of
	 * renumbering, and are more preferred.  However, since there is no
	 * standard naming convention and APIs to deal with the names.  Thus,
	 * we only handle the case of link-local addresses, for which we use
	 * interface names as link names, assuming one to one mapping between
	 * interfaces and links.
	 */
#ifdef ISC_PLATFORM_HAVEIFNAMETOINDEX
	if (IN6_IS_ADDR_LINKLOCAL(in6) &&
	    (ifid = if_nametoindex((const char *)scopename)) != 0)
		zone = (isc_uint32_t)ifid;
	else {
#endif
		llz = isc_string_touint64(scopename, &ep, 10);
		if (ep == scopename)
			return (ISC_R_FAILURE);

		/* check overflow */
		zone = (isc_uint32_t)(llz & 0xffffffffUL);
		if (zone != llz)
			return (ISC_R_FAILURE);
#ifdef ISC_PLATFORM_HAVEIFNAMETOINDEX
	}
#endif

	*zoneid = zone;
	return (ISC_R_SUCCESS);
}
