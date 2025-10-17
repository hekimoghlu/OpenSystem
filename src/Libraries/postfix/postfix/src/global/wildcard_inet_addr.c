/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <msg.h>
#include <inet_addr_list.h>
#include <inet_addr_host.h>

/* Global library. */

#include <wildcard_inet_addr.h>

/* Application-specific. */

static INET_ADDR_LIST wild_addr_list;

static void wildcard_inet_addr_init(INET_ADDR_LIST *addr_list)
{
    inet_addr_list_init(addr_list);
    if (inet_addr_host(addr_list, "") == 0)
	msg_fatal("could not get list of wildcard addresses");
}

/* wildcard_inet_addr_list - return list of addresses */

INET_ADDR_LIST *wildcard_inet_addr_list(void)
{
    if (wild_addr_list.used == 0)
	wildcard_inet_addr_init(&wild_addr_list);

    return (&wild_addr_list);
}
