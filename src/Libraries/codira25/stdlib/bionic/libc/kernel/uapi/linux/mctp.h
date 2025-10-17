/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#ifndef __UAPI_MCTP_H
#define __UAPI_MCTP_H
#include <linux/types.h>
#include <linux/socket.h>
#include <linux/netdevice.h>
typedef __u8 mctp_eid_t;
struct mctp_addr {
  mctp_eid_t s_addr;
};
struct sockaddr_mctp {
  __kernel_sa_family_t smctp_family;
  __u16 __smctp_pad0;
  unsigned int smctp_network;
  struct mctp_addr smctp_addr;
  __u8 smctp_type;
  __u8 smctp_tag;
  __u8 __smctp_pad1;
};
struct sockaddr_mctp_ext {
  struct sockaddr_mctp smctp_base;
  int smctp_ifindex;
  __u8 smctp_halen;
  __u8 __smctp_pad0[3];
  __u8 smctp_haddr[MAX_ADDR_LEN];
};
#define MCTP_NET_ANY 0x0
#define MCTP_ADDR_NULL 0x00
#define MCTP_ADDR_ANY 0xff
#define MCTP_TAG_MASK 0x07
#define MCTP_TAG_OWNER 0x08
#define MCTP_TAG_PREALLOC 0x10
#define MCTP_OPT_ADDR_EXT 1
#define SIOCMCTPALLOCTAG (SIOCPROTOPRIVATE + 0)
#define SIOCMCTPDROPTAG (SIOCPROTOPRIVATE + 1)
#define SIOCMCTPALLOCTAG2 (SIOCPROTOPRIVATE + 2)
#define SIOCMCTPDROPTAG2 (SIOCPROTOPRIVATE + 3)
struct mctp_ioc_tag_ctl {
  mctp_eid_t peer_addr;
  __u8 tag;
  __u16 flags;
};
struct mctp_ioc_tag_ctl2 {
  unsigned int net;
  mctp_eid_t peer_addr;
  mctp_eid_t local_addr;
  __u16 flags;
  __u8 tag;
};
#endif
