/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
/* $Id: lwaddr.c,v 1.10 2008/01/11 23:46:56 tbox Exp $ */

/*! \file */

#include <config.h>

#include <string.h>

#include <isc/result.h>
#include <isc/netaddr.h>
#include <isc/sockaddr.h>

#include <lwres/lwres.h>

#include <named/lwaddr.h>

/*%
 * Convert addresses from lwres to isc format.
 */
isc_result_t
lwaddr_netaddr_fromlwresaddr(isc_netaddr_t *na, lwres_addr_t *la) {
	if (la->family != LWRES_ADDRTYPE_V4 && la->family != LWRES_ADDRTYPE_V6)
		return (ISC_R_FAMILYNOSUPPORT);

	if (la->family == LWRES_ADDRTYPE_V4) {
		struct in_addr ina;
		memmove(&ina.s_addr, la->address, 4);
		isc_netaddr_fromin(na, &ina);
	} else {
		struct in6_addr ina6;
		memmove(&ina6.s6_addr, la->address, 16);
		isc_netaddr_fromin6(na, &ina6);
	}
	return (ISC_R_SUCCESS);
}

isc_result_t
lwaddr_sockaddr_fromlwresaddr(isc_sockaddr_t *sa, lwres_addr_t *la,
			      in_port_t port)
{
	isc_netaddr_t na;
	isc_result_t result;

	result = lwaddr_netaddr_fromlwresaddr(&na, la);
	if (result != ISC_R_SUCCESS)
		return (result);
	isc_sockaddr_fromnetaddr(sa, &na, port);
	return (ISC_R_SUCCESS);
}

/*%
 * Convert addresses from isc to lwres format.
 */

isc_result_t
lwaddr_lwresaddr_fromnetaddr(lwres_addr_t *la, isc_netaddr_t *na) {
	if (na->family != AF_INET && na->family != AF_INET6)
		return (ISC_R_FAMILYNOSUPPORT);

	if (na->family == AF_INET) {
		la->family = LWRES_ADDRTYPE_V4;
		la->length = 4;
		memmove(la->address, &na->type.in, 4);
	} else {
		la->family = LWRES_ADDRTYPE_V6;
		la->length = 16;
		memmove(la->address, &na->type.in6, 16);
	}
	return (ISC_R_SUCCESS);
}

isc_result_t
lwaddr_lwresaddr_fromsockaddr(lwres_addr_t *la, isc_sockaddr_t *sa) {
	isc_netaddr_t na;
	isc_netaddr_fromsockaddr(&na, sa);
	return (lwaddr_lwresaddr_fromnetaddr(la, &na));
}
