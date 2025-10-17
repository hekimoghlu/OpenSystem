/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#ifndef _MISCFS_SPECFS_IO_COMPRESSION_STATS_H_
#define _MISCFS_SPECFS_IO_COMPRESSION_STATS_H_

#include <sys/buf_internal.h>
#include <sys/vnode.h>

void io_compression_stats_init(void);
void io_compression_stats(buf_t bp);

#define IO_COMPRESSION_STATS_DEFAULT_BLOCK_SIZE (4 * 1024)
#define IO_COMPRESSION_STATS_MIN_BLOCK_SIZE (4 * 1024)
#define IO_COMPRESSION_STATS_MAX_BLOCK_SIZE (1024 * 1024 * 1024)

#if IO_COMPRESSION_STATS_DEBUG
#define io_compression_stats_dbg(fmt, ...) \
	printf("%s: " fmt "\n", __func__, ## __VA_ARGS__)
#else
#define io_compression_stats_dbg(fmt, ...)
#endif

/* iocs_store_buffer: Buffer that captures the stats of vnode being reclaimed */
struct iocs_store_buffer {
	void*                   buffer;
	uint32_t                current_position;
	uint32_t                marked_point;
};

#define IOCS_STORE_BUFFER_NUM_SLOTS 10000
#define IOCS_STORE_BUFFER_SIZE (IOCS_STORE_BUFFER_NUM_SLOTS * (sizeof(struct iocs_store_buffer_entry)))

/* Notify user when the buffer is 80% full */
#define IOCS_STORE_BUFFER_NOTIFY_AT ((IOCS_STORE_BUFFER_SIZE * 8) / 10)

/* Wait for the buffer to be 10% more full before notifying again */
#define IOCS_STORE_BUFFER_NOTIFICATION_INTERVAL (IOCS_STORE_BUFFER_SIZE / 10)

#endif
