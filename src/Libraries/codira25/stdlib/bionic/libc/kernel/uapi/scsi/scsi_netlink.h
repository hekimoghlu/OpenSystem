/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef SCSI_NETLINK_H
#define SCSI_NETLINK_H
#include <linux/netlink.h>
#include <linux/types.h>
#define SCSI_TRANSPORT_MSG NLMSG_MIN_TYPE + 1
#define SCSI_NL_GRP_FC_EVENTS (1 << 2)
#define SCSI_NL_GRP_CNT 3
struct scsi_nl_hdr {
  __u8 version;
  __u8 transport;
  __u16 magic;
  __u16 msgtype;
  __u16 msglen;
} __attribute__((aligned(sizeof(__u64))));
#define SCSI_NL_VERSION 1
#define SCSI_NL_MAGIC 0xA1B2
#define SCSI_NL_TRANSPORT 0
#define SCSI_NL_TRANSPORT_FC 1
#define SCSI_NL_MAX_TRANSPORTS 2
#define SCSI_NL_SHOST_VENDOR 0x0001
#define SCSI_NL_MSGALIGN(len) (((len) + 7) & ~7)
struct scsi_nl_host_vendor_msg {
  struct scsi_nl_hdr snlh;
  __u64 vendor_id;
  __u16 host_no;
  __u16 vmsg_datalen;
} __attribute__((aligned(sizeof(__u64))));
#define SCSI_NL_VID_TYPE_SHIFT 56
#define SCSI_NL_VID_TYPE_MASK ((__u64) 0xFF << SCSI_NL_VID_TYPE_SHIFT)
#define SCSI_NL_VID_TYPE_PCI ((__u64) 0x01 << SCSI_NL_VID_TYPE_SHIFT)
#define SCSI_NL_VID_ID_MASK (~SCSI_NL_VID_TYPE_MASK)
#define INIT_SCSI_NL_HDR(hdr,t,mtype,mlen) { (hdr)->version = SCSI_NL_VERSION; (hdr)->transport = t; (hdr)->magic = SCSI_NL_MAGIC; (hdr)->msgtype = mtype; (hdr)->msglen = mlen; }
#endif
