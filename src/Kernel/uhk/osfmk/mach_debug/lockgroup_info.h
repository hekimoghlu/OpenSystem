/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
 *	File:	mach/lockgroup_info.h
 *
 *	Definitions for host_lockgroup_info call.
 */

#ifndef _MACH_DEBUG_LOCKGROUP_INFO_H_
#define _MACH_DEBUG_LOCKGROUP_INFO_H_

#include <mach/mach_types.h>

#define LOCKGROUP_MAX_NAME      64

#define LOCKGROUP_ATTR_STAT     0x01ULL

typedef struct lockgroup_info {
	char            lockgroup_name[LOCKGROUP_MAX_NAME];
	uint64_t        lockgroup_attr;
	uint64_t        lock_spin_cnt;
	uint64_t        lock_spin_util_cnt;
	uint64_t        lock_spin_held_cnt;
	uint64_t        lock_spin_miss_cnt;
	uint64_t        lock_spin_held_max;
	uint64_t        lock_spin_held_cum;
	uint64_t        lock_mtx_cnt;
	uint64_t        lock_mtx_util_cnt;
	uint64_t        lock_mtx_held_cnt;
	uint64_t        lock_mtx_miss_cnt;
	uint64_t        lock_mtx_wait_cnt;
	uint64_t        lock_mtx_held_max;
	uint64_t        lock_mtx_held_cum;
	uint64_t        lock_mtx_wait_max;
	uint64_t        lock_mtx_wait_cum;
	uint64_t        lock_rw_cnt;
	uint64_t        lock_rw_util_cnt;
	uint64_t        lock_rw_held_cnt;
	uint64_t        lock_rw_miss_cnt;
	uint64_t        lock_rw_wait_cnt;
	uint64_t        lock_rw_held_max;
	uint64_t        lock_rw_held_cum;
	uint64_t        lock_rw_wait_max;
	uint64_t        lock_rw_wait_cum;
} lockgroup_info_t;

typedef lockgroup_info_t *lockgroup_info_array_t;

#endif  /* _MACH_DEBUG_LOCKGROUP_INFO_H_ */
