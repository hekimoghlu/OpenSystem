/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#ifndef _UAPILINUX_PHONET_H
#define _UAPILINUX_PHONET_H
#include <linux/types.h>
#include <linux/socket.h>
#define PN_PROTO_TRANSPORT 0
#define PN_PROTO_PHONET 1
#define PN_PROTO_PIPE 2
#define PHONET_NPROTO 3
#define PNPIPE_ENCAP 1
#define PNPIPE_IFINDEX 2
#define PNPIPE_HANDLE 3
#define PNPIPE_INITSTATE 4
#define PNADDR_ANY 0
#define PNADDR_BROADCAST 0xFC
#define PNPORT_RESOURCE_ROUTING 0
#define PNPIPE_ENCAP_NONE 0
#define PNPIPE_ENCAP_IP 1
#define SIOCPNGETOBJECT (SIOCPROTOPRIVATE + 0)
#define SIOCPNENABLEPIPE (SIOCPROTOPRIVATE + 13)
#define SIOCPNADDRESOURCE (SIOCPROTOPRIVATE + 14)
#define SIOCPNDELRESOURCE (SIOCPROTOPRIVATE + 15)
struct phonethdr {
  __u8 pn_rdev;
  __u8 pn_sdev;
  __u8 pn_res;
  __be16 pn_length;
  __u8 pn_robj;
  __u8 pn_sobj;
} __attribute__((packed));
struct phonetmsg {
  __u8 pn_trans_id;
  __u8 pn_msg_id;
  union {
    struct {
      __u8 pn_submsg_id;
      __u8 pn_data[5];
    } base;
    struct {
      __u16 pn_e_res_id;
      __u8 pn_e_submsg_id;
      __u8 pn_e_data[3];
    } ext;
  } pn_msg_u;
};
#define PN_COMMON_MESSAGE 0xF0
#define PN_COMMGR 0x10
#define PN_PREFIX 0xE0
#define pn_submsg_id pn_msg_u.base.pn_submsg_id
#define pn_e_submsg_id pn_msg_u.ext.pn_e_submsg_id
#define pn_e_res_id pn_msg_u.ext.pn_e_res_id
#define pn_data pn_msg_u.base.pn_data
#define pn_e_data pn_msg_u.ext.pn_e_data
#define PN_COMM_SERVICE_NOT_IDENTIFIED_RESP 0x01
#define PN_COMM_ISA_ENTITY_NOT_REACHABLE_RESP 0x14
#define pn_orig_msg_id pn_data[0]
#define pn_status pn_data[1]
#define pn_e_orig_msg_id pn_e_data[0]
#define pn_e_status pn_e_data[1]
struct sockaddr_pn {
  __kernel_sa_family_t spn_family;
  __u8 spn_obj;
  __u8 spn_dev;
  __u8 spn_resource;
  __u8 spn_zero[sizeof(struct sockaddr) - sizeof(__kernel_sa_family_t) - 3];
} __attribute__((packed));
#define PN_DEV_PC 0x10
#endif
