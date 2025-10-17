/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
/* $Id: getaddresses.h,v 1.11 2009/01/17 23:47:42 tbox Exp $ */

#ifndef BIND9_GETADDRESSES_H
#define BIND9_GETADDRESSES_H 1

/*! \file bind9/getaddresses.h */

#include <isc/lang.h>
#include <isc/types.h>

#include <isc/net.h>

ISC_LANG_BEGINDECLS

isc_result_t
bind9_getaddresses(const char *hostname, in_port_t port,
		   isc_sockaddr_t *addrs, int addrsize, int *addrcount);
/*%<
 * Use the system resolver to get the addresses associated with a hostname.
 * If successful, the number of addresses found is returned in 'addrcount'.
 * If a hostname lookup is performed and addresses of an unknown family is
 * seen, it is ignored.  If more than 'addrsize' addresses are seen, the
 * first 'addrsize' are returned and the remainder silently truncated.
 *
 * This routine may block.  If called by a program using the isc_app
 * framework, it should be surrounded by isc_app_block()/isc_app_unblock().
 *
 *  Requires:
 *\li	'hostname' is not NULL.
 *\li	'addrs' is not NULL.
 *\li	'addrsize' > 0
 *\li	'addrcount' is not NULL.
 *
 *
 * Returns:
 *\li	#ISC_R_SUCCESS
 *\li	#ISC_R_NOTFOUND
 *\li	#ISC_R_NOFAMILYSUPPORT - 'hostname' is an IPv6 address, and IPv6 is
 *		not supported.
 */

ISC_LANG_ENDDECLS

#endif /* BIND9_GETADDRESSES_H */
