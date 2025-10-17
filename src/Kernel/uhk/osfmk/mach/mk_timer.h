/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
 * Copyright (c) 2000 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 * 31 August 2000 (debo)
 *  Created.
 */

#ifndef _MACH_MK_TIMER_H_
#define _MACH_MK_TIMER_H_

#include <mach/mach_time.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

extern mach_port_name_t mk_timer_create(void);

extern kern_return_t    mk_timer_destroy(
	mach_port_name_t        name);

extern kern_return_t    mk_timer_arm(
	mach_port_name_t        name,
	uint64_t                expire_time);

extern kern_return_t    mk_timer_cancel(
	mach_port_name_t        name,
	uint64_t               *result_time);

/* mk_timer_flags */
#define MK_TIMER_NORMAL         (0)
#define MK_TIMER_CRITICAL       (1)

extern kern_return_t    mk_timer_arm_leeway(
	mach_port_name_t        name,
	uint64_t                mk_timer_flags,
	uint64_t                mk_timer_expire_time,
	uint64_t                mk_timer_leeway);

#pragma pack(4)
typedef struct mk_timer_expire_msg {
	mach_msg_header_t       header;
	uint64_t                unused[3];
} mk_timer_expire_msg_t;
#pragma pack()

__END_DECLS

#endif /* _MACH_MK_TIMER_H_ */
