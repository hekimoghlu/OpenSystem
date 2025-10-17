/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
#ifndef __LINUX_IF_ADDR_H
#define __LINUX_IF_ADDR_H
#include <linux/types.h>
#include <linux/netlink.h>
struct ifaddrmsg {
  __u8 ifa_family;
  __u8 ifa_prefixlen;
  __u8 ifa_flags;
  __u8 ifa_scope;
  __u32 ifa_index;
};
enum {
  IFA_UNSPEC,
  IFA_ADDRESS,
  IFA_LOCAL,
  IFA_LABEL,
  IFA_BROADCAST,
  IFA_ANYCAST,
  IFA_CACHEINFO,
  IFA_MULTICAST,
  IFA_FLAGS,
  IFA_RT_PRIORITY,
  IFA_TARGET_NETNSID,
  IFA_PROTO,
  __IFA_MAX,
};
#define IFA_MAX (__IFA_MAX - 1)
#define IFA_F_SECONDARY 0x01
#define IFA_F_TEMPORARY IFA_F_SECONDARY
#define IFA_F_NODAD 0x02
#define IFA_F_OPTIMISTIC 0x04
#define IFA_F_DADFAILED 0x08
#define IFA_F_HOMEADDRESS 0x10
#define IFA_F_DEPRECATED 0x20
#define IFA_F_TENTATIVE 0x40
#define IFA_F_PERMANENT 0x80
#define IFA_F_MANAGETEMPADDR 0x100
#define IFA_F_NOPREFIXROUTE 0x200
#define IFA_F_MCAUTOJOIN 0x400
#define IFA_F_STABLE_PRIVACY 0x800
struct ifa_cacheinfo {
  __u32 ifa_prefered;
  __u32 ifa_valid;
  __u32 cstamp;
  __u32 tstamp;
};
#define IFA_RTA(r) ((struct rtattr *) (((char *) (r)) + NLMSG_ALIGN(sizeof(struct ifaddrmsg))))
#define IFA_PAYLOAD(n) NLMSG_PAYLOAD(n, sizeof(struct ifaddrmsg))
#define IFAPROT_UNSPEC 0
#define IFAPROT_KERNEL_LO 1
#define IFAPROT_KERNEL_RA 2
#define IFAPROT_KERNEL_LL 3
#endif
