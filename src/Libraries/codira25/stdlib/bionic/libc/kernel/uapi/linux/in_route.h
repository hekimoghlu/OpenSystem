/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#ifndef _LINUX_IN_ROUTE_H
#define _LINUX_IN_ROUTE_H
#define RTCF_DEAD RTNH_F_DEAD
#define RTCF_ONLINK RTNH_F_ONLINK
#define RTCF_NOPMTUDISC RTM_F_NOPMTUDISC
#define RTCF_NOTIFY 0x00010000
#define RTCF_DIRECTDST 0x00020000
#define RTCF_REDIRECTED 0x00040000
#define RTCF_TPROXY 0x00080000
#define RTCF_FAST 0x00200000
#define RTCF_MASQ 0x00400000
#define RTCF_SNAT 0x00800000
#define RTCF_DOREDIRECT 0x01000000
#define RTCF_DIRECTSRC 0x04000000
#define RTCF_DNAT 0x08000000
#define RTCF_BROADCAST 0x10000000
#define RTCF_MULTICAST 0x20000000
#define RTCF_REJECT 0x40000000
#define RTCF_LOCAL 0x80000000
#define RTCF_NAT (RTCF_DNAT | RTCF_SNAT)
#define RT_TOS(tos) ((tos) & IPTOS_TOS_MASK)
#endif
