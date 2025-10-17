/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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
#ifndef __LINUX_TC_SKBEDIT_H
#define __LINUX_TC_SKBEDIT_H
#include <linux/pkt_cls.h>
#define SKBEDIT_F_PRIORITY 0x1
#define SKBEDIT_F_QUEUE_MAPPING 0x2
#define SKBEDIT_F_MARK 0x4
#define SKBEDIT_F_PTYPE 0x8
#define SKBEDIT_F_MASK 0x10
#define SKBEDIT_F_INHERITDSFIELD 0x20
#define SKBEDIT_F_TXQ_SKBHASH 0x40
struct tc_skbedit {
  tc_gen;
};
enum {
  TCA_SKBEDIT_UNSPEC,
  TCA_SKBEDIT_TM,
  TCA_SKBEDIT_PARMS,
  TCA_SKBEDIT_PRIORITY,
  TCA_SKBEDIT_QUEUE_MAPPING,
  TCA_SKBEDIT_MARK,
  TCA_SKBEDIT_PAD,
  TCA_SKBEDIT_PTYPE,
  TCA_SKBEDIT_MASK,
  TCA_SKBEDIT_FLAGS,
  TCA_SKBEDIT_QUEUE_MAPPING_MAX,
  __TCA_SKBEDIT_MAX
};
#define TCA_SKBEDIT_MAX (__TCA_SKBEDIT_MAX - 1)
#endif
