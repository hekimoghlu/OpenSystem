/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#ifndef _UAPI_LINUX_IPV6_ROUTE_H
#define _UAPI_LINUX_IPV6_ROUTE_H
#include <linux/types.h>
#include <linux/in6.h>
#define RTF_DEFAULT 0x00010000
#define RTF_ALLONLINK 0x00020000
#define RTF_ADDRCONF 0x00040000
#define RTF_PREFIX_RT 0x00080000
#define RTF_ANYCAST 0x00100000
#define RTF_NONEXTHOP 0x00200000
#define RTF_EXPIRES 0x00400000
#define RTF_ROUTEINFO 0x00800000
#define RTF_CACHE 0x01000000
#define RTF_FLOW 0x02000000
#define RTF_POLICY 0x04000000
#define RTF_PREF(pref) ((pref) << 27)
#define RTF_PREF_MASK 0x18000000
#define RTF_PCPU 0x40000000
#define RTF_LOCAL 0x80000000
struct in6_rtmsg {
  struct in6_addr rtmsg_dst;
  struct in6_addr rtmsg_src;
  struct in6_addr rtmsg_gateway;
  __u32 rtmsg_type;
  __u16 rtmsg_dst_len;
  __u16 rtmsg_src_len;
  __u32 rtmsg_metric;
  unsigned long rtmsg_info;
  __u32 rtmsg_flags;
  int rtmsg_ifindex;
};
#define RTMSG_NEWDEVICE 0x11
#define RTMSG_DELDEVICE 0x12
#define RTMSG_NEWROUTE 0x21
#define RTMSG_DELROUTE 0x22
#define IP6_RT_PRIO_USER 1024
#define IP6_RT_PRIO_ADDRCONF 256
#endif
