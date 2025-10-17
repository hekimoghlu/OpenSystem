/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#ifndef __LINUX_TC_GATE_H
#define __LINUX_TC_GATE_H
#include <linux/pkt_cls.h>
struct tc_gate {
  tc_gen;
};
enum {
  TCA_GATE_ENTRY_UNSPEC,
  TCA_GATE_ENTRY_INDEX,
  TCA_GATE_ENTRY_GATE,
  TCA_GATE_ENTRY_INTERVAL,
  TCA_GATE_ENTRY_IPV,
  TCA_GATE_ENTRY_MAX_OCTETS,
  __TCA_GATE_ENTRY_MAX,
};
#define TCA_GATE_ENTRY_MAX (__TCA_GATE_ENTRY_MAX - 1)
enum {
  TCA_GATE_ONE_ENTRY_UNSPEC,
  TCA_GATE_ONE_ENTRY,
  __TCA_GATE_ONE_ENTRY_MAX,
};
#define TCA_GATE_ONE_ENTRY_MAX (__TCA_GATE_ONE_ENTRY_MAX - 1)
enum {
  TCA_GATE_UNSPEC,
  TCA_GATE_TM,
  TCA_GATE_PARMS,
  TCA_GATE_PAD,
  TCA_GATE_PRIORITY,
  TCA_GATE_ENTRY_LIST,
  TCA_GATE_BASE_TIME,
  TCA_GATE_CYCLE_TIME,
  TCA_GATE_CYCLE_TIME_EXT,
  TCA_GATE_FLAGS,
  TCA_GATE_CLOCKID,
  __TCA_GATE_MAX,
};
#define TCA_GATE_MAX (__TCA_GATE_MAX - 1)
#endif
