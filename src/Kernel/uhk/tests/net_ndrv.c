/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
 * net_ndrv.c
 * - test ndrv socket
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/errno.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <net/if_media.h>
#include <net/if_types.h>
#include <net/if_dl.h>
#include <net/dlil.h>
#include <net/ndrv.h>
#include <net/ethernet.h>
#include <sys/sockio.h>
#include <fcntl.h>
#include <stdbool.h>
#include <TargetConditionals.h>
#include <darwintest_utils.h>

static const struct ether_addr multicast_one = {
	{ 0x01, 0x80, 0xc2, 0x00, 0x00, 0x01 }
};

static const struct ether_addr multicast_two = {
	{ 0x01, 0x80, 0xc2, 0x00, 0x00, 0x02 }
};

static void
ndrv_socket_do_multicast(int s, const struct ether_addr * multiaddr,
    bool add)
{
	struct sockaddr_dl      dl;
	int                     status;


	bzero(&dl, sizeof(dl));
	dl.sdl_len = sizeof(dl);
	dl.sdl_family = AF_LINK;
	dl.sdl_type = IFT_ETHER;
	dl.sdl_nlen = 0;
	dl.sdl_alen = sizeof(*multiaddr);
	bcopy(multiaddr, dl.sdl_data, sizeof(*multiaddr));
	status = setsockopt(s, SOL_NDRVPROTO,
	    add ? NDRV_ADDMULTICAST : NDRV_DELMULTICAST,
	    &dl, dl.sdl_len);
	T_ASSERT_POSIX_SUCCESS(status,
	    "setsockopt(NDRV_%sMULTICAST)",
	    add ? "ADD" : "DEL");
}

static void
ndrv_socket_add_multicast(int s, const struct ether_addr * multiaddr)
{
	ndrv_socket_do_multicast(s, multiaddr, true);
}

static void
ndrv_socket_remove_multicast(int s, const struct ether_addr * multiaddr)
{
	ndrv_socket_do_multicast(s, multiaddr, false);
}

static int
ndrv_socket_open(const char * ifname)
{
	struct sockaddr_ndrv    ndrv;
	int                     s;
	int                     status;

	s = socket(AF_NDRV, SOCK_RAW, 0);
	T_ASSERT_POSIX_SUCCESS(s, "socket(AF_NDRV, SOCK_RAW, 0)");
	bzero(&ndrv, sizeof(ndrv));
	strlcpy((char *)ndrv.snd_name, ifname, sizeof(ndrv.snd_name));
	ndrv.snd_len = sizeof(ndrv);
	ndrv.snd_family = AF_NDRV;
	status = bind(s, (struct sockaddr *)&ndrv, sizeof(ndrv));
	T_ASSERT_POSIX_SUCCESS(status, "bind ndrv socket");
	return s;
}

static void
ndrv_socket_multicast_add_remove(const char * ifname)
{
	int                     s;

	/* test for rdar://99667160 */
	s = ndrv_socket_open(ifname);
	ndrv_socket_add_multicast(s, &multicast_one);
	ndrv_socket_add_multicast(s, &multicast_two);
	ndrv_socket_remove_multicast(s, &multicast_one);
	close(s);
}

T_DECL(ndrv_socket_multicast_add_remove,
    "ndrv socket multicast add remove",
    T_META_ASROOT(true), T_META_TAG_VM_PREFERRED)
{
	ndrv_socket_multicast_add_remove("lo0");
}
