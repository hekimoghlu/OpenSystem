/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#ifndef _LINUX_ATMARP_H
#define _LINUX_ATMARP_H
#include <linux/types.h>
#include <linux/atmapi.h>
#include <linux/atmioc.h>
#define ATMARP_RETRY_DELAY 30
#define ATMARP_MAX_UNRES_PACKETS 5
#define ATMARPD_CTRL _IO('a', ATMIOC_CLIP + 1)
#define ATMARP_MKIP _IO('a', ATMIOC_CLIP + 2)
#define ATMARP_SETENTRY _IO('a', ATMIOC_CLIP + 3)
#define ATMARP_ENCAP _IO('a', ATMIOC_CLIP + 5)
enum atmarp_ctrl_type {
  act_invalid,
  act_need,
  act_up,
  act_down,
  act_change
};
struct atmarp_ctrl {
  enum atmarp_ctrl_type type;
  int itf_num;
  __be32 ip;
};
#endif
