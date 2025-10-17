/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#ifndef _UAPI__IP_SET_HASH_H
#define _UAPI__IP_SET_HASH_H
#include <linux/netfilter/ipset/ip_set.h>
enum {
  IPSET_ERR_HASH_FULL = IPSET_ERR_TYPE_SPECIFIC,
  IPSET_ERR_HASH_ELEM,
  IPSET_ERR_INVALID_PROTO,
  IPSET_ERR_MISSING_PROTO,
  IPSET_ERR_HASH_RANGE_UNSUPPORTED,
  IPSET_ERR_HASH_RANGE,
};
#endif
