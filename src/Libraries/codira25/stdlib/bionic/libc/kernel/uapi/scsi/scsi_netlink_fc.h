/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#ifndef SCSI_NETLINK_FC_H
#define SCSI_NETLINK_FC_H
#include <linux/types.h>
#include <scsi/scsi_netlink.h>
#define FC_NL_ASYNC_EVENT 0x0100
#define FC_NL_MSGALIGN(len) (((len) + 7) & ~7)
struct fc_nl_event {
  struct scsi_nl_hdr snlh;
  __u64 seconds;
  __u64 vendor_id;
  __u16 host_no;
  __u16 event_datalen;
  __u32 event_num;
  __u32 event_code;
  union {
    __u32 event_data;
    __DECLARE_FLEX_ARRAY(__u8, event_data_flex);
  };
} __attribute__((aligned(sizeof(__u64))));
#endif
