/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#ifndef _ARPA_INET_H_
#define _ARPA_INET_H_

#include <sys/cdefs.h>

#include <netinet/in.h>
#include <stdint.h>
#include <sys/types.h>

__BEGIN_DECLS

in_addr_t inet_addr(const char* _Nonnull __s);
int inet_aton(const char* _Nonnull __s, struct in_addr* _Nullable __addr);
in_addr_t inet_lnaof(struct in_addr __addr);
struct in_addr inet_makeaddr(in_addr_t __net, in_addr_t __host);
in_addr_t inet_netof(struct in_addr __addr);
in_addr_t inet_network(const char* _Nonnull __s);
char* _Nonnull inet_ntoa(struct in_addr __addr);
const char* _Nullable inet_ntop(int __af, const void* _Nonnull __src, char* _Nonnull __dst, socklen_t __size);
unsigned int inet_nsap_addr(const char* _Nonnull __ascii, unsigned char* _Nonnull __binary, int __n);
char* _Nonnull inet_nsap_ntoa(int __binary_length, const unsigned char* _Nonnull __binary, char* _Nullable __ascii);
int inet_pton(int __af, const char* _Nonnull __src, void* _Nonnull __dst);

__END_DECLS

#endif
