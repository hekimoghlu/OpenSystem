/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
 * inetroute.h
 * - get a list of internet network routes (includes default route)
 */

/*
 * Modification History
 *
 * Dieter Siegmund (dieter@apple.com) Tue Jul 14 11:33:50 PDT 1998
 * - created
 */

#ifndef _S_BOOTPLIB_INETROUTE_H
#define _S_BOOTPLIB_INETROUTE_H

#include <CoreFoundation/CFString.h>

typedef struct {
    struct in_addr		dest;
    struct in_addr		mask;
    union {
	struct sockaddr_dl	link;
	struct sockaddr_in	inet;
    } gateway;
} inetroute_t;

typedef struct {
    int			count;
    inetroute_t * 	list;
    int			def_index;
} inetroute_list_t;

inetroute_list_t *	inetroute_list_init();
void			inetroute_list_free(inetroute_list_t * * list);
struct in_addr *	inetroute_default(inetroute_list_t * list_p);
void			inetroute_list_print(inetroute_list_t * list_p);
void			inetroute_list_print_cfstr(CFMutableStringRef str,
						   inetroute_list_t * list_p);
#endif /* _S_BOOTPLIB_INETROUTE_H */
