/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#ifndef _SYS_COALITION_PRIVATE_H_
#define _SYS_COALITION_PRIVATE_H_

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS

#define COALITION_POLICY_ENTITLEMENT "com.apple.private.coalition-policy"

__enum_decl(coalition_policy_flavor_t, uint32_t, {
	COALITION_POLICY_SUPPRESS = 1,
});

__enum_decl(coalition_policy_suppress_t, uint32_t, {
	COALITION_POLICY_SUPPRESS_NONE = 0,
	COALITION_POLICY_SUPPRESS_DARWIN_BG = 1,
});

#ifndef KERNEL
/* Userspace syscall prototypes */
int coalition_policy_set(uint64_t cid, coalition_policy_flavor_t flavor, uint32_t value);
int coalition_policy_get(uint64_t cid, coalition_policy_flavor_t flavor);
#endif /* #ifndef KERNEL */

__END_DECLS

#endif /* _SYS_COALITION_PRIVATE_H_ */
