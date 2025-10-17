/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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
#ifndef __NSI_STR_ID__
#define __NSI_STR_ID__

#ifdef  KERNEL_PRIVATE

#include <sys/types.h>
#include <sys/kernel_types.h>
#include <sys/queue.h>

struct net_str_id_entry {
	SLIST_ENTRY(net_str_id_entry) nsi_next;
	uint32_t nsi_flags;
	uint32_t nsi_id;
	uint32_t nsi_length;
	char nsi_string[__counted_by(nsi_length)];
};

enum {
	NSI_MBUF_TAG    = 0,
	NSI_VENDOR_CODE = 1,
	NSI_IF_FAM_ID   = 2,
	NSI_MAX_KIND
};

extern void net_str_id_first_last(u_int32_t *, u_int32_t *, u_int32_t);

extern errno_t net_str_id_find_internal(const char *, u_int32_t *, u_int32_t, int);

extern void net_str_id_init(void);

#endif /* KERNEL_PRIVATE */

#endif /* __NSI_STR_ID__ */
