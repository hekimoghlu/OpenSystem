/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#include <sys/cdefs.h>
#include <sys/types.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/fcntl.h>
#include <sys/coalition.h>

/* Syscall entry points */
int __coalition(uint32_t operation, uint64_t *cid, uint32_t flags);
int __coalition_info(uint32_t operation, uint64_t *cid, void *buffer, size_t *bufsize);
int __coalition_ledger(uint32_t operation, uint64_t *cid, void *buffer, size_t *bufsize);

int
coalition_create(uint64_t *cid_out, uint32_t flags)
{
	return __coalition(COALITION_OP_CREATE, cid_out, flags);
}

int
coalition_terminate(uint64_t cid, uint32_t flags)
{
	return __coalition(COALITION_OP_TERMINATE, &cid, flags);
}

int
coalition_reap(uint64_t cid, uint32_t flags)
{
	return __coalition(COALITION_OP_REAP, &cid, flags);
}

int
coalition_info_resource_usage(uint64_t cid, struct coalition_resource_usage *cru, size_t sz)
{
	return __coalition_info(COALITION_INFO_RESOURCE_USAGE, &cid, cru, &sz);
}

int
coalition_info_debug_info(uint64_t cid, struct coalinfo_debuginfo *cru, size_t sz)
{
	return __coalition_info(COALITION_INFO_GET_DEBUG_INFO, &cid, cru, &sz);
}

int
coalition_info_set_name(uint64_t cid, const char *name, size_t size)
{
	return __coalition_info(COALITION_INFO_SET_NAME, &cid, (void *)name, &size);
}

int
coalition_info_set_efficiency(uint64_t cid, uint64_t flags)
{
	size_t size = sizeof(flags);
	return __coalition_info(COALITION_INFO_SET_EFFICIENCY, &cid, (void *)&flags, &size);
}

int
coalition_ledger_set_logical_writes_limit(uint64_t cid, int64_t limit)
{
	size_t size = sizeof(limit);
	return __coalition_ledger(COALITION_LEDGER_SET_LOGICAL_WRITES_LIMIT, &cid, (void *)&limit, &size);
}
