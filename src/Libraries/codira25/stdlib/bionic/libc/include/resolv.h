/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#ifndef _RESOLV_H_
#define _RESOLV_H_

#include <sys/cdefs.h>

#include <sys/param.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <arpa/nameser.h>
#include <netinet/in.h>

__BEGIN_DECLS

#define b64_ntop __b64_ntop
int b64_ntop(u_char const* _Nonnull __src, size_t __src_size, char* _Nonnull __dst, size_t __dst_size);
#define b64_pton __b64_pton
int b64_pton(char const* _Nonnull __src, u_char* _Nonnull __dst, size_t __dst_size);

#define dn_comp __dn_comp
int dn_comp(const char* _Nonnull __src, u_char* _Nonnull __dst, int __dst_size, u_char* _Nullable * _Nullable __dn_ptrs , u_char* _Nullable * _Nullable __last_dn_ptr);

int dn_expand(const u_char* _Nonnull __msg, const u_char* _Nonnull __eom, const u_char* _Nonnull __src, char* _Nonnull __dst, int __dst_size);

#define p_class __p_class
const char* _Nonnull p_class(int __class);
#define p_type __p_type
const char* _Nonnull p_type(int __type);

int res_init(void);
int res_mkquery(int __opcode, const char* _Nonnull __domain_name, int __class, int __type, const u_char* _Nullable __data, int __data_size, const u_char* _Nullable __new_rr_in, u_char* _Nonnull __buf, int __buf_size);
int res_query(const char* _Nonnull __name, int __class, int __type, u_char* _Nonnull __answer, int __answer_size);
int res_search(const char* _Nonnull __name, int __class, int __type, u_char* _Nonnull __answer, int __answer_size);

#define res_randomid __res_randomid

#if __BIONIC_AVAILABILITY_GUARD(29)
u_int __res_randomid(void) __INTRODUCED_IN(29);
#endif /* __BIONIC_AVAILABILITY_GUARD(29) */


__END_DECLS

#endif
