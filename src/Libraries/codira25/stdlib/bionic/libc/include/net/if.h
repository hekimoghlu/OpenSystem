/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#ifndef _NET_IF_H_
#define _NET_IF_H_

#include <sys/cdefs.h>

#include <sys/socket.h>
#include <linux/if.h>

#ifndef IF_NAMESIZE
#define IF_NAMESIZE IFNAMSIZ
#endif

__BEGIN_DECLS

struct if_nameindex {
  unsigned if_index;
  char* _Nullable if_name;
};

char* _Nullable if_indextoname(unsigned __index, char* _Nonnull __buf);
unsigned if_nametoindex(const char* _Nonnull __name);

#if __BIONIC_AVAILABILITY_GUARD(24)
struct if_nameindex* _Nullable if_nameindex(void) __INTRODUCED_IN(24);
void if_freenameindex(struct if_nameindex* _Nullable __ptr) __INTRODUCED_IN(24);
#endif /* __BIONIC_AVAILABILITY_GUARD(24) */


__END_DECLS

#endif
