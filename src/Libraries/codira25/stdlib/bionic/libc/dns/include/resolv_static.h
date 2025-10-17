/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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

#include <netdb.h>
#include <sys/cdefs.h>

/* this structure contains all the variables that were declared
 * 'static' in the original NetBSD resolver code.
 *
 * this caused vast amounts of crashes and memory corruptions
 * when the resolver was being used by multiple threads.
 *
 * (note: the OpenBSD/FreeBSD resolver has similar 'issues')
 */

#define	MAXALIASES	35
#define	MAXADDRS	35

__BEGIN_DECLS

struct res_static {
  char* h_addr_ptrs[MAXADDRS + 1];
  char* host_aliases[MAXALIASES];
  char hostbuf[8 * 1024];
  u_int32_t host_addr[16 / sizeof(u_int32_t)]; /* IPv4 or IPv6 */
  FILE* hostf;
  int stayopen;
  const char* servent_ptr;
  struct servent servent;
  struct hostent host;
};

struct res_static* __res_get_static(void);

__END_DECLS
