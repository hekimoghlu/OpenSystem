/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#pragma once

#include <sys/cdefs.h>

#include <endian.h>
#include <netinet/in6.h>
#include <sys/socket.h>

#include <linux/in.h>
#include <linux/in6.h>
#include <linux/ipv6.h>
#include <linux/socket.h>

__BEGIN_DECLS

#define INET_ADDRSTRLEN 16

typedef uint16_t in_port_t;

int bindresvport(int __fd, struct sockaddr_in* _Nullable __sin);

#if __ANDROID_API__ >= 24
extern const struct in6_addr in6addr_any __INTRODUCED_IN(24);
extern const struct in6_addr in6addr_loopback __INTRODUCED_IN(24);
#else
static const struct in6_addr in6addr_any = IN6ADDR_ANY_INIT;
static const struct in6_addr in6addr_loopback = IN6ADDR_LOOPBACK_INIT;
#endif

__END_DECLS
