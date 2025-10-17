/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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

//
//  hfs_alloc_trace.h
//  hfs
//
//  Created by Chris Suter on 8/19/15.
//
//

#ifndef hfs_alloc_trace_h
#define hfs_alloc_trace_h

#include <sys/types.h>
#include <stdbool.h>

enum {
	HFS_ALLOC_BACKTRACE_LEN = 4,
};

#pragma pack(push, 8)

struct hfs_alloc_trace_info {
	int entry_count;
	bool more;
	struct hfs_alloc_info_entry {
		uint64_t ptr;
		uint64_t sequence;
		uint64_t size;
		uint64_t backtrace[HFS_ALLOC_BACKTRACE_LEN];
	} entries[];
};

#pragma pack(pop)

#endif /* hfs_alloc_trace_h */
