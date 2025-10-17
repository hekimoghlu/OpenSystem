/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
 * @OSF_COPYRIGHT@
 */

#ifndef _MACH_SYNC_POLICY_H_
#define _MACH_SYNC_POLICY_H_

typedef int sync_policy_t;

/*
 *	These options define the wait ordering of the synchronizers
 */
#define SYNC_POLICY_FIFO                0x0
#define SYNC_POLICY_FIXED_PRIORITY      0x1
#define SYNC_POLICY_REVERSED            0x2
#define SYNC_POLICY_ORDER_MASK          0x3
#define SYNC_POLICY_LIFO                (SYNC_POLICY_FIFO|SYNC_POLICY_REVERSED)

#if KERNEL_PRIVATE

#define SYNC_POLICY_PREPOST             0x4 /* obsolete but kexts use it */

#endif /* KERNEL_PRIVATE */
#ifdef XNU_KERNEL_PRIVATE

/* SYNC_POLICY_FIXED_PRIORITY is no longer supported */
#define SYNC_POLICY_USER_MASK \
	(SYNC_POLICY_FIFO | SYNC_POLICY_LIFO | SYNC_POLICY_PREPOST)

#define SYNC_POLICY_INIT_LOCKED         0x08

#endif  /* XNU_KERNEL_PRIVATE */

#endif  /* _MACH_SYNC_POLICY_H_ */
