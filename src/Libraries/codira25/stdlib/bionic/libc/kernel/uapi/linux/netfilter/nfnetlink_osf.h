/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#ifndef _NF_OSF_H
#define _NF_OSF_H
#include <linux/types.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#define MAXGENRELEN 32
#define NF_OSF_GENRE (1 << 0)
#define NF_OSF_TTL (1 << 1)
#define NF_OSF_LOG (1 << 2)
#define NF_OSF_INVERT (1 << 3)
#define NF_OSF_LOGLEVEL_ALL 0
#define NF_OSF_LOGLEVEL_FIRST 1
#define NF_OSF_LOGLEVEL_ALL_KNOWN 2
#define NF_OSF_TTL_TRUE 0
#define NF_OSF_TTL_LESS 1
#define NF_OSF_TTL_NOCHECK 2
#define NF_OSF_FLAGMASK (NF_OSF_GENRE | NF_OSF_TTL | NF_OSF_LOG | NF_OSF_INVERT)
struct nf_osf_wc {
  __u32 wc;
  __u32 val;
};
struct nf_osf_opt {
  __u16 kind, length;
  struct nf_osf_wc wc;
};
struct nf_osf_info {
  char genre[MAXGENRELEN];
  __u32 len;
  __u32 flags;
  __u32 loglevel;
  __u32 ttl;
};
struct nf_osf_user_finger {
  struct nf_osf_wc wss;
  __u8 ttl, df;
  __u16 ss, mss;
  __u16 opt_num;
  char genre[MAXGENRELEN];
  char version[MAXGENRELEN];
  char subtype[MAXGENRELEN];
  struct nf_osf_opt opt[MAX_IPOPTLEN];
};
struct nf_osf_nlmsg {
  struct nf_osf_user_finger f;
  struct iphdr ip;
  struct tcphdr tcp;
};
enum iana_options {
  OSFOPT_EOL = 0,
  OSFOPT_NOP,
  OSFOPT_MSS,
  OSFOPT_WSO,
  OSFOPT_SACKP,
  OSFOPT_SACK,
  OSFOPT_ECHO,
  OSFOPT_ECHOREPLY,
  OSFOPT_TS,
  OSFOPT_POCP,
  OSFOPT_POSP,
  OSFOPT_EMPTY = 255,
};
enum nf_osf_window_size_options {
  OSF_WSS_PLAIN = 0,
  OSF_WSS_MSS,
  OSF_WSS_MTU,
  OSF_WSS_MODULO,
  OSF_WSS_MAX,
};
enum nf_osf_attr_type {
  OSF_ATTR_UNSPEC,
  OSF_ATTR_FINGER,
  OSF_ATTR_MAX,
};
enum nf_osf_msg_types {
  OSF_MSG_ADD,
  OSF_MSG_REMOVE,
  OSF_MSG_MAX,
};
#endif
