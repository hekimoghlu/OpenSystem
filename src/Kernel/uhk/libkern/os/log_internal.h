/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#ifndef log_internal_h
#define log_internal_h

#include <os/log_private.h>

typedef struct {
	firehose_tracepoint_id_u    lp_ftid;
	uint64_t                    lp_timestamp;
	uint16_t                    lp_pub_data_size;
	uint16_t                    lp_data_size;
	firehose_stream_t           lp_stream;
} log_payload_s, *log_payload_t;

bool log_payload_send(log_payload_t, const void *, bool);
bool os_log_subsystem_id_valid(uint16_t);

#endif /* log_internal */
