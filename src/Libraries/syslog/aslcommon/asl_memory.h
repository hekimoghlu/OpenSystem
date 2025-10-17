/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
#ifndef __ASL_MEMORY_H__
#define __ASL_MEMORY_H__
#include <stdint.h>
#include <asl.h>
#include <dispatch/dispatch.h>

typedef struct
{
	uint32_t hash;
	uint32_t refcount;
	char *str;
} mem_string_t;

typedef struct
{
	uint64_t mid;
	uint64_t time;
	uint64_t os_activity_id;
	mem_string_t *host;
	mem_string_t *sender;
	mem_string_t *sender_mach_uuid;
	mem_string_t *facility;
	mem_string_t *message;
	mem_string_t *refproc;
	mem_string_t *session;
	mem_string_t **kvlist;
	uint32_t nano;
	uint32_t pid;
	uint32_t uid;
	uint32_t gid;
	uint32_t ruid;
	uint32_t rgid;
	uint32_t refpid;
	uint32_t kvcount;
	uint16_t flags;
	uint8_t level;
	uint8_t unused_0;
} mem_record_t;

typedef struct
{
	mem_string_t **string_cache;
	mem_record_t **record;
	mem_record_t *buffer_record;
	uint32_t string_count;
	uint32_t record_count;
	uint32_t record_first;
	size_t max_string_mem;
	size_t curr_string_mem;
	dispatch_queue_t queue;
} asl_memory_t;

uint32_t asl_memory_open(uint32_t max_records, size_t max_str_mem, asl_memory_t **s);
uint32_t asl_memory_close(asl_memory_t *s);
uint32_t asl_memory_statistics(asl_memory_t *s, asl_msg_t **msg);

uint32_t asl_memory_save(asl_memory_t *s, asl_msg_t *msg, uint64_t *mid);
uint32_t asl_memory_fetch(asl_memory_t *s, uint64_t mid, asl_msg_t **msg, int32_t ruid, int32_t rgid);

uint32_t asl_memory_match(asl_memory_t *s, asl_msg_list_t *query, asl_msg_list_t **res, uint64_t *last_id, uint64_t start_id, uint32_t count, int32_t direction, int32_t ruid, int32_t rgid);
uint32_t asl_memory_match_restricted_uuid(asl_memory_t *s, asl_msg_list_t *query, asl_msg_list_t **res, uint64_t *last_id, uint64_t start_id, uint32_t count, uint32_t duration, int32_t direction, int32_t ruid, int32_t rgid, const char *uuid_str);

#endif /* __ASL_MEMORY_H__ */
