/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#ifndef _XT_TPROXY_H
#define _XT_TPROXY_H
#include <linux/types.h>
#include <linux/netfilter.h>
struct xt_tproxy_target_info {
  __u32 mark_mask;
  __u32 mark_value;
  __be32 laddr;
  __be16 lport;
};
struct xt_tproxy_target_info_v1 {
  __u32 mark_mask;
  __u32 mark_value;
  union nf_inet_addr laddr;
  __be16 lport;
};
#endif
