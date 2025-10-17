/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#ifndef _NETINET_IN6_H
#define _NETINET_IN6_H

#include <sys/cdefs.h>

#include <linux/in6.h>

#define IN6_IS_ADDR_UNSPECIFIED(a) \
  ((((a)->s6_addr32[0]) == 0) && \
   (((a)->s6_addr32[1]) == 0) && \
   (((a)->s6_addr32[2]) == 0) && \
   (((a)->s6_addr32[3]) == 0))

#define IN6_IS_ADDR_LOOPBACK(a) \
  ((((a)->s6_addr32[0]) == 0) && \
   (((a)->s6_addr32[1]) == 0) && \
   (((a)->s6_addr32[2]) == 0) && \
   (((a)->s6_addr32[3]) == ntohl(1)))

#define IN6_IS_ADDR_V4COMPAT(a) \
  ((((a)->s6_addr32[0]) == 0) && \
   (((a)->s6_addr32[1]) == 0) && \
   (((a)->s6_addr32[2]) == 0) && \
   (((a)->s6_addr32[3]) != 0) && (((a)->s6_addr32[3]) != ntohl(1)))

#define IN6_IS_ADDR_V4MAPPED(a) \
  ((((a)->s6_addr32[0]) == 0) && \
   (((a)->s6_addr32[1]) == 0) && \
   (((a)->s6_addr32[2]) == ntohl(0x0000ffff)))

#define __bionic_s6_addr(a) __BIONIC_CAST(reinterpret_cast, const uint8_t*, a)

#define IN6_IS_ADDR_LINKLOCAL(a) \
  ((__bionic_s6_addr(a)[0] == 0xfe) && ((__bionic_s6_addr(a)[1] & 0xc0) == 0x80))

#define IN6_IS_ADDR_SITELOCAL(a) \
  ((__bionic_s6_addr(a)[0] == 0xfe) && ((__bionic_s6_addr(a)[1] & 0xc0) == 0xc0))

#define IN6_IS_ADDR_MULTICAST(a) (__bionic_s6_addr(a)[0] == 0xff)

#define IN6_IS_ADDR_ULA(a) ((__bionic_s6_addr(a)[0] & 0xfe) == 0xfc)

#define IPV6_ADDR_SCOPE_NODELOCAL       0x01
#define IPV6_ADDR_SCOPE_INTFACELOCAL    0x01
#define IPV6_ADDR_SCOPE_LINKLOCAL       0x02
#define IPV6_ADDR_SCOPE_SITELOCAL       0x05
#define IPV6_ADDR_SCOPE_ORGLOCAL        0x08
#define IPV6_ADDR_SCOPE_GLOBAL          0x0e

#define IPV6_ADDR_MC_SCOPE(a) (__bionic_s6_addr(a)[1] & 0x0f)

#define IN6_IS_ADDR_MC_NODELOCAL(a) \
  (IN6_IS_ADDR_MULTICAST(a) && (IPV6_ADDR_MC_SCOPE(a) == IPV6_ADDR_SCOPE_NODELOCAL))
#define IN6_IS_ADDR_MC_LINKLOCAL(a) \
  (IN6_IS_ADDR_MULTICAST(a) && (IPV6_ADDR_MC_SCOPE(a) == IPV6_ADDR_SCOPE_LINKLOCAL))
#define IN6_IS_ADDR_MC_SITELOCAL(a) \
  (IN6_IS_ADDR_MULTICAST(a) && (IPV6_ADDR_MC_SCOPE(a) == IPV6_ADDR_SCOPE_SITELOCAL))
#define IN6_IS_ADDR_MC_ORGLOCAL(a) \
  (IN6_IS_ADDR_MULTICAST(a) && (IPV6_ADDR_MC_SCOPE(a) == IPV6_ADDR_SCOPE_ORGLOCAL))
#define IN6_IS_ADDR_MC_GLOBAL(a) \
  (IN6_IS_ADDR_MULTICAST(a) && (IPV6_ADDR_MC_SCOPE(a) == IPV6_ADDR_SCOPE_GLOBAL))

#define IN6_ARE_ADDR_EQUAL(a, b) \
  (__builtin_memcmp(&(a)->s6_addr[0], &(b)->s6_addr[0], sizeof(struct in6_addr)) == 0)

#define INET6_ADDRSTRLEN 46

#define IPV6_JOIN_GROUP IPV6_ADD_MEMBERSHIP
#define IPV6_LEAVE_GROUP IPV6_DROP_MEMBERSHIP

#define IN6ADDR_ANY_INIT {{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}}
#define IN6ADDR_LOOPBACK_INIT {{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}}

#define ipv6mr_interface ipv6mr_ifindex

#endif /* _NETINET_IN6_H */
