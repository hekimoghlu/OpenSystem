/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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
#ifndef __LINUX_TC_EM_CMP_H
#define __LINUX_TC_EM_CMP_H
#include <linux/types.h>
#include <linux/pkt_cls.h>
struct tcf_em_cmp {
  __u32 val;
  __u32 mask;
  __u16 off;
  __u8 align : 4;
  __u8 flags : 4;
  __u8 layer : 4;
  __u8 opnd : 4;
};
enum {
  TCF_EM_ALIGN_U8 = 1,
  TCF_EM_ALIGN_U16 = 2,
  TCF_EM_ALIGN_U32 = 4
};
#define TCF_EM_CMP_TRANS 1
#endif
