/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef _NET_FLOWHASH_H_
#define _NET_FLOWHASH_H_

#include <sys/types.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * If 32-bit hash value is too large, use this macro to truncate
 * it to n-bit; masking is a faster operation than modulus.
 */
#define HASHMASK(n)     ((1UL << (n)) - 1)

/*
 * Returns 32-bit hash value.  Hashes which are capable of returning
 * more bits currently have their results truncated to 32-bit.
 */
typedef u_int32_t net_flowhash_fn_t(const void *__sized_by(len) key, u_int32_t len, const u_int32_t);

extern net_flowhash_fn_t *net_flowhash;
extern net_flowhash_fn_t net_flowhash_mh3_x86_32;
extern net_flowhash_fn_t net_flowhash_mh3_x64_128;
extern net_flowhash_fn_t net_flowhash_jhash;
#ifdef  __cplusplus
}
#endif

#endif /* _NET_FLOWHASH_H_ */
