/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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
#ifndef __ASM_GENERIC_IPCBUF_H
#define __ASM_GENERIC_IPCBUF_H
#include <linux/posix_types.h>
struct ipc64_perm {
  __kernel_key_t key;
  __kernel_uid32_t uid;
  __kernel_gid32_t gid;
  __kernel_uid32_t cuid;
  __kernel_gid32_t cgid;
  __kernel_mode_t mode;
  unsigned char __pad1[4 - sizeof(__kernel_mode_t)];
  unsigned short seq;
  unsigned short __pad2;
  __kernel_ulong_t __unused1;
  __kernel_ulong_t __unused2;
};
#endif
