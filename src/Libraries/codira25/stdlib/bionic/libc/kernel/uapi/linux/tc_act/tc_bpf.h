/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#ifndef __LINUX_TC_BPF_H
#define __LINUX_TC_BPF_H
#include <linux/pkt_cls.h>
struct tc_act_bpf {
  tc_gen;
};
enum {
  TCA_ACT_BPF_UNSPEC,
  TCA_ACT_BPF_TM,
  TCA_ACT_BPF_PARMS,
  TCA_ACT_BPF_OPS_LEN,
  TCA_ACT_BPF_OPS,
  TCA_ACT_BPF_FD,
  TCA_ACT_BPF_NAME,
  TCA_ACT_BPF_PAD,
  TCA_ACT_BPF_TAG,
  TCA_ACT_BPF_ID,
  __TCA_ACT_BPF_MAX,
};
#define TCA_ACT_BPF_MAX (__TCA_ACT_BPF_MAX - 1)
#endif
