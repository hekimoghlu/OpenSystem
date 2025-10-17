/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#pragma once

/**
 * @file netinet/igmp.h
 * @brief Internet Group Management Protocol (IGMP).
 */

#include <sys/cdefs.h>
#include <netinet/in.h>

#include <linux/igmp.h>

/**
 * The uapi type is called `igmphdr`,
 * doesn't have the `igmp_` prefix on each field,
 * and uses a `__be32` for the group address.
 *
 * This is the type that BSDs and musl/glibc expose to userspace.
 */
struct igmp {
  uint8_t igmp_type;
  uint8_t igmp_code;
  uint16_t igmp_cksum;
  struct in_addr igmp_group;
};

/** Commonly-used BSD synonym for the Linux constant. */
#define IGMP_MEMBERSHIP_QUERY IGMP_HOST_MEMBERSHIP_QUERY
