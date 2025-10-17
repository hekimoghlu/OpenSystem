/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#ifndef libtrace_trace_internal_h
#define libtrace_trace_internal_h

#include <os/log.h>
#include <uuid/uuid.h>
#include <kern/assert.h>
#include <firehose/firehose_types_private.h>

__BEGIN_DECLS

OS_ALWAYS_INLINE
inline uint32_t
_os_trace_offset(const void *dso, const void *addr, _firehose_tracepoint_flags_activity_t flags __unused)
{
	assert((uintptr_t)addr >= (uintptr_t)dso);
	return (uint32_t) ((uintptr_t)addr - (uintptr_t)dso);
}

bool
_os_trace_addr_in_text_segment(const void *dso, const void *addr);

__END_DECLS

#endif
