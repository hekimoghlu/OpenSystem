/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

#ifndef __PLATFORM_INTROSPECTION_INTERNAL_H__
#define __PLATFORM_INTROSPECTION_INTERNAL_H__


// Task Allocations (with 4K vm_page_size)
//
// Page 1                              Page 2
// +----------------+ <---- Head ----- +----------------+
// | Header         | ----- Next ----> | Header         |  ----- Next ----> ...
// +----------------+                  +----------------+
// | Meta Data      |                  | Thread[126]    |
// +----------------+                  +----------------+
// | Thread[0]      |                  | Thread[127]    |
// +----------------+                  +----------------+
// ~ ...            ~                  ~ ...            ~
// +----------------+                  +----------------+
// | Thread[125]    |                  | Thread[252]    |
// +----------------+                  +----------------+
//

// 32 bytes
struct platform_task_header_s {
	// Linkage to other platform task pages.
	platform_task_t head;
	platform_task_t next;
#ifdef __LP64__
	uint64_t _reserved[2];
#else
	uint32_t _reserved[6];
#endif
};

// 32 bytes
struct platform_task_metadata_s {
	platform_thread_t cursor;
	task_t port;
	uint32_t gen;
	uint32_t is_64_bit:1, unused_flags:31;
#ifdef __LP64__
	uint32_t _reserved[3];
#else
	uint32_t _reserved[4];
#endif
};

// 32 bytes
struct platform_thread_s {
	int32_t refcnt;
	uint32_t gen;
	platform_thread_id_t unique_id;
	mach_vm_address_t pthread_addr;
	thread_act_t act;
	uint32_t _reserved;
};

struct platform_task_s {
	struct platform_task_header_s header;
	union {
		struct platform_task_metadata_s metadata;
		struct platform_thread_s threads[127];
	};
};

#endif // __PLATFORM_INTROSPECTION_INTERNAL_H__
