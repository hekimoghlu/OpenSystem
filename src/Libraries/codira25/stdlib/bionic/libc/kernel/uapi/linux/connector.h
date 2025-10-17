/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#ifndef _UAPI__CONNECTOR_H
#define _UAPI__CONNECTOR_H
#include <linux/types.h>
#define CN_IDX_PROC 0x1
#define CN_VAL_PROC 0x1
#define CN_IDX_CIFS 0x2
#define CN_VAL_CIFS 0x1
#define CN_W1_IDX 0x3
#define CN_W1_VAL 0x1
#define CN_IDX_V86D 0x4
#define CN_VAL_V86D_UVESAFB 0x1
#define CN_IDX_BB 0x5
#define CN_DST_IDX 0x6
#define CN_DST_VAL 0x1
#define CN_IDX_DM 0x7
#define CN_VAL_DM_USERSPACE_LOG 0x1
#define CN_IDX_DRBD 0x8
#define CN_VAL_DRBD 0x1
#define CN_KVP_IDX 0x9
#define CN_KVP_VAL 0x1
#define CN_VSS_IDX 0xA
#define CN_VSS_VAL 0x1
#define CN_NETLINK_USERS 11
#define CONNECTOR_MAX_MSG_SIZE 16384
struct cb_id {
  __u32 idx;
  __u32 val;
};
struct cn_msg {
  struct cb_id id;
  __u32 seq;
  __u32 ack;
  __u16 len;
  __u16 flags;
  __u8 data[];
};
#endif
