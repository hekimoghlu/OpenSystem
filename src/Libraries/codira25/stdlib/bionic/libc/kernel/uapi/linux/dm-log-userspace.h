/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#ifndef __DM_LOG_USERSPACE_H__
#define __DM_LOG_USERSPACE_H__
#include <linux/types.h>
#include <linux/dm-ioctl.h>
#define DM_ULOG_CTR 1
#define DM_ULOG_DTR 2
#define DM_ULOG_PRESUSPEND 3
#define DM_ULOG_POSTSUSPEND 4
#define DM_ULOG_RESUME 5
#define DM_ULOG_GET_REGION_SIZE 6
#define DM_ULOG_IS_CLEAN 7
#define DM_ULOG_IN_SYNC 8
#define DM_ULOG_FLUSH 9
#define DM_ULOG_MARK_REGION 10
#define DM_ULOG_CLEAR_REGION 11
#define DM_ULOG_GET_RESYNC_WORK 12
#define DM_ULOG_SET_REGION_SYNC 13
#define DM_ULOG_GET_SYNC_COUNT 14
#define DM_ULOG_STATUS_INFO 15
#define DM_ULOG_STATUS_TABLE 16
#define DM_ULOG_IS_REMOTE_RECOVERING 17
#define DM_ULOG_REQUEST_MASK 0xFF
#define DM_ULOG_REQUEST_TYPE(request_type) (DM_ULOG_REQUEST_MASK & (request_type))
#define DM_ULOG_REQUEST_VERSION 3
struct dm_ulog_request {
  __u64 luid;
  char uuid[DM_UUID_LEN];
  char padding[3];
  __u32 version;
  __s32 error;
  __u32 seq;
  __u32 request_type;
  __u32 data_size;
  char data[];
};
#endif
