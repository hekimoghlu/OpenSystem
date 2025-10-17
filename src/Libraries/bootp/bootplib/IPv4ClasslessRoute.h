/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
 * IPv4ClasslessRoute.h
 * - handle IPv4 route lists in DHCP options
 */

/*
 * Modification History
 *
 * June 5, 2014			Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_IPV4CLASSLESSROUTE_H
#define _S_IPV4CLASSLESSROUTE_H


#include <stdbool.h>
#include <stdint.h>
#include <netinet/in.h>
#include <CoreFoundation/CFArray.h>
#include "symbol_scope.h"

typedef struct {
    struct in_addr	dest;
    int			prefix_length;
    struct in_addr	gate;		/* 0.0.0.0 => direct to interface */
} IPv4ClasslessRoute, * IPv4ClasslessRouteRef;


uint8_t *
IPv4ClasslessRouteListBufferCreate(IPv4ClasslessRouteRef list, int list_count,
				   uint8_t * buffer, int * buffer_size);

IPv4ClasslessRouteRef
IPv4ClasslessRouteListCreate(const uint8_t * buffer, int buffer_size,
			     int * list_count);

IPv4ClasslessRouteRef
IPv4ClasslessRouteListGetDefault(IPv4ClasslessRouteRef list, int list_count);

IPv4ClasslessRouteRef
IPv4ClasslessRouteListCreateWithArray(CFArrayRef string_list,
				      int * ret_count);

#endif /* _S_IPV4CLASSLESSROUTE_H */

