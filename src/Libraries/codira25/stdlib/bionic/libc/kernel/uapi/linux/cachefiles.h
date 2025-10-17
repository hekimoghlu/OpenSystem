/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#ifndef _LINUX_CACHEFILES_H
#define _LINUX_CACHEFILES_H
#include <linux/types.h>
#include <linux/ioctl.h>
#define CACHEFILES_MSG_MAX_SIZE 1024
enum cachefiles_opcode {
  CACHEFILES_OP_OPEN,
  CACHEFILES_OP_CLOSE,
  CACHEFILES_OP_READ,
};
struct cachefiles_msg {
  __u32 msg_id;
  __u32 opcode;
  __u32 len;
  __u32 object_id;
  __u8 data[];
};
struct cachefiles_open {
  __u32 volume_key_size;
  __u32 cookie_key_size;
  __u32 fd;
  __u32 flags;
  __u8 data[];
};
struct cachefiles_read {
  __u64 off;
  __u64 len;
};
#define CACHEFILES_IOC_READ_COMPLETE _IOW(0x98, 1, int)
#endif
