/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#ifndef _UAPI_NF_CONNTRACK_TCP_H
#define _UAPI_NF_CONNTRACK_TCP_H
#include <linux/types.h>
enum tcp_conntrack {
  TCP_CONNTRACK_NONE,
  TCP_CONNTRACK_SYN_SENT,
  TCP_CONNTRACK_SYN_RECV,
  TCP_CONNTRACK_ESTABLISHED,
  TCP_CONNTRACK_FIN_WAIT,
  TCP_CONNTRACK_CLOSE_WAIT,
  TCP_CONNTRACK_LAST_ACK,
  TCP_CONNTRACK_TIME_WAIT,
  TCP_CONNTRACK_CLOSE,
  TCP_CONNTRACK_LISTEN,
#define TCP_CONNTRACK_SYN_SENT2 TCP_CONNTRACK_LISTEN
  TCP_CONNTRACK_MAX,
  TCP_CONNTRACK_IGNORE,
  TCP_CONNTRACK_RETRANS,
  TCP_CONNTRACK_UNACK,
  TCP_CONNTRACK_TIMEOUT_MAX
};
#define IP_CT_TCP_FLAG_WINDOW_SCALE 0x01
#define IP_CT_TCP_FLAG_SACK_PERM 0x02
#define IP_CT_TCP_FLAG_CLOSE_INIT 0x04
#define IP_CT_TCP_FLAG_BE_LIBERAL 0x08
#define IP_CT_TCP_FLAG_DATA_UNACKNOWLEDGED 0x10
#define IP_CT_TCP_FLAG_MAXACK_SET 0x20
#define IP_CT_EXP_CHALLENGE_ACK 0x40
#define IP_CT_TCP_SIMULTANEOUS_OPEN 0x80
struct nf_ct_tcp_flags {
  __u8 flags;
  __u8 mask;
};
#endif
