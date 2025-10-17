/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#ifndef _UAPI__LINUX_ATALK_H__
#define _UAPI__LINUX_ATALK_H__
#include <linux/types.h>
#include <asm/byteorder.h>
#include <linux/socket.h>
#define ATPORT_FIRST 1
#define ATPORT_RESERVED 128
#define ATPORT_LAST 254
#define ATADDR_ANYNET (__u16) 0
#define ATADDR_ANYNODE (__u8) 0
#define ATADDR_ANYPORT (__u8) 0
#define ATADDR_BCAST (__u8) 255
#define DDP_MAXSZ 587
#define DDP_MAXHOPS 15
#define SIOCATALKDIFADDR (SIOCPROTOPRIVATE + 0)
struct atalk_addr {
  __be16 s_net;
  __u8 s_node;
};
struct sockaddr_at {
  __kernel_sa_family_t sat_family;
  __u8 sat_port;
  struct atalk_addr sat_addr;
  char sat_zero[8];
};
struct atalk_netrange {
  __u8 nr_phase;
  __be16 nr_firstnet;
  __be16 nr_lastnet;
};
#endif
