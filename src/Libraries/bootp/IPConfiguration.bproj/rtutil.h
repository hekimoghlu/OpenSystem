/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
/*
 * rtutil.h
 * - routing table routines
 */

/* 
 * Modification History
 *
 * June 23, 2009	Dieter Siegmund (dieter@apple.com)
 * - split out from ipconfigd.c
 */

#ifndef _S_RTUTIL_H
#define _S_RTUTIL_H

#include <net/route.h>
#include <netinet/in.h>
#include <mach/boolean.h>

boolean_t
subnet_route_add(struct in_addr gateway, struct in_addr netaddr, 
		 struct in_addr netmask, const char * ifname);

void
flush_routes(int if_index, const struct in_addr ip,
	     const struct in_addr broadcast);

#endif /* _S_RTUTIL_H */
