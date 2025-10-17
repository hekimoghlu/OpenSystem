/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#ifndef __LINUX_TC_SKBMOD_H
#define __LINUX_TC_SKBMOD_H
#include <linux/pkt_cls.h>
#define SKBMOD_F_DMAC 0x1
#define SKBMOD_F_SMAC 0x2
#define SKBMOD_F_ETYPE 0x4
#define SKBMOD_F_SWAPMAC 0x8
#define SKBMOD_F_ECN 0x10
struct tc_skbmod {
  tc_gen;
  __u64 flags;
};
enum {
  TCA_SKBMOD_UNSPEC,
  TCA_SKBMOD_TM,
  TCA_SKBMOD_PARMS,
  TCA_SKBMOD_DMAC,
  TCA_SKBMOD_SMAC,
  TCA_SKBMOD_ETYPE,
  TCA_SKBMOD_PAD,
  __TCA_SKBMOD_MAX
};
#define TCA_SKBMOD_MAX (__TCA_SKBMOD_MAX - 1)
#endif
