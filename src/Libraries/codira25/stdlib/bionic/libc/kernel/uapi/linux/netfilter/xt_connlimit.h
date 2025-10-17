/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifndef _XT_CONNLIMIT_H
#define _XT_CONNLIMIT_H
#include <linux/types.h>
#include <linux/netfilter.h>
struct xt_connlimit_data;
enum {
  XT_CONNLIMIT_INVERT = 1 << 0,
  XT_CONNLIMIT_DADDR = 1 << 1,
};
struct xt_connlimit_info {
  union {
    union nf_inet_addr mask;
    union {
      __be32 v4_mask;
      __be32 v6_mask[4];
    };
  };
  unsigned int limit;
  __u32 flags;
  struct nf_conncount_data * data __attribute__((aligned(8)));
};
#endif
