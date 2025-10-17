/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
 * Copyright (C) 2000 WIDE Project.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _NETINET6_SCOPE6_VAR_H_
#define _NETINET6_SCOPE6_VAR_H_
#include <sys/appleapiopts.h>

/*
 * 16 is correspondent to 4bit multicast scope field.
 * i.e. from node-local to global with some reserved/unassigned types.
 */
#define SCOPE6_ID_MAX   16

#ifdef BSD_KERNEL_PRIVATE

struct scope6_id {
	u_int32_t s6id_list[SCOPE6_ID_MAX];
};

extern int in6_embedded_scope;
extern int in6_embedded_scope_debug;

#define IN6_NULL_IF_EMBEDDED_SCOPE(_var) (in6_embedded_scope ? NULL : (_var))

extern void scope6_ifattach(struct ifnet *);
extern void scope6_setdefault(struct ifnet *);
extern u_int32_t scope6_in6_addrscope(struct in6_addr *);
extern u_int32_t scope6_addr2default(struct in6_addr *);
extern int sa6_embedscope(struct sockaddr_in6 *, int, uint32_t *);
extern int sa6_recoverscope(struct sockaddr_in6 *, boolean_t);
extern int in6_setscope(struct in6_addr *, struct ifnet *, u_int32_t *);
extern int in6_clearscope(struct in6_addr *);
extern void rtkey_to_sa6(struct rtentry *, struct sockaddr_in6 *);
extern void rtgw_to_sa6(struct rtentry *, struct sockaddr_in6 *);
extern bool in6_are_addr_equal_scoped(const struct in6_addr *, const struct in6_addr *,
    uint32_t, uint32_t);
extern bool in6_are_masked_addr_scope_equal(const struct in6_addr *, uint32_t, const struct in6_addr *, uint32_t, const struct in6_addr *);

extern void in6_verify_ifscope(const struct in6_addr *, uint32_t);

#endif /* BSD_KERNEL_PRIVATE */
#endif /* _NETINET6_SCOPE6_VAR_H_ */
