/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#ifndef _LINUX_SELINUX_NETLINK_H
#define _LINUX_SELINUX_NETLINK_H
#include <linux/types.h>
#define SELNL_MSG_BASE 0x10
enum {
  SELNL_MSG_SETENFORCE = SELNL_MSG_BASE,
  SELNL_MSG_POLICYLOAD,
  SELNL_MSG_MAX
};
#define SELNL_GRP_NONE 0x00000000
#define SELNL_GRP_AVC 0x00000001
#define SELNL_GRP_ALL 0xffffffff
enum selinux_nlgroups {
  SELNLGRP_NONE,
#define SELNLGRP_NONE SELNLGRP_NONE
  SELNLGRP_AVC,
#define SELNLGRP_AVC SELNLGRP_AVC
  __SELNLGRP_MAX
};
#define SELNLGRP_MAX (__SELNLGRP_MAX - 1)
struct selnl_msg_setenforce {
  __s32 val;
};
struct selnl_msg_policyload {
  __u32 seqno;
};
#endif
