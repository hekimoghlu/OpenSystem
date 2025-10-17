/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#ifndef _UAPI__DLM_PLOCK_DOT_H__
#define _UAPI__DLM_PLOCK_DOT_H__
#include <linux/types.h>
#define DLM_PLOCK_MISC_NAME "dlm_plock"
#define DLM_PLOCK_VERSION_MAJOR 1
#define DLM_PLOCK_VERSION_MINOR 2
#define DLM_PLOCK_VERSION_PATCH 0
enum {
  DLM_PLOCK_OP_LOCK = 1,
  DLM_PLOCK_OP_UNLOCK,
  DLM_PLOCK_OP_GET,
  DLM_PLOCK_OP_CANCEL,
};
#define DLM_PLOCK_FL_CLOSE 1
struct dlm_plock_info {
  __u32 version[3];
  __u8 optype;
  __u8 ex;
  __u8 wait;
  __u8 flags;
  __u32 pid;
  __s32 nodeid;
  __s32 rv;
  __u32 fsid;
  __u64 number;
  __u64 start;
  __u64 end;
  __u64 owner;
};
#endif
