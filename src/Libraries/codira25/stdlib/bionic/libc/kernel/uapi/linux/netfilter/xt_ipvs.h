/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#ifndef _XT_IPVS_H
#define _XT_IPVS_H
#include <linux/types.h>
#include <linux/netfilter.h>
enum {
  XT_IPVS_IPVS_PROPERTY = 1 << 0,
  XT_IPVS_PROTO = 1 << 1,
  XT_IPVS_VADDR = 1 << 2,
  XT_IPVS_VPORT = 1 << 3,
  XT_IPVS_DIR = 1 << 4,
  XT_IPVS_METHOD = 1 << 5,
  XT_IPVS_VPORTCTL = 1 << 6,
  XT_IPVS_MASK = (1 << 7) - 1,
  XT_IPVS_ONCE_MASK = XT_IPVS_MASK & ~XT_IPVS_IPVS_PROPERTY
};
struct xt_ipvs_mtinfo {
  union nf_inet_addr vaddr, vmask;
  __be16 vport;
  __u8 l4proto;
  __u8 fwd_method;
  __be16 vportctl;
  __u8 invert;
  __u8 bitmask;
};
#endif
