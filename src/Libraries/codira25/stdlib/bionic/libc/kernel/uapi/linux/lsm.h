/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#ifndef _UAPI_LINUX_LSM_H
#define _UAPI_LINUX_LSM_H
#include <linux/stddef.h>
#include <linux/types.h>
#include <linux/unistd.h>
struct lsm_ctx {
  __u64 id;
  __u64 flags;
  __u64 len;
  __u64 ctx_len;
  __u8 ctx[] __counted_by(ctx_len);
};
#define LSM_ID_UNDEF 0
#define LSM_ID_CAPABILITY 100
#define LSM_ID_SELINUX 101
#define LSM_ID_SMACK 102
#define LSM_ID_TOMOYO 103
#define LSM_ID_APPARMOR 104
#define LSM_ID_YAMA 105
#define LSM_ID_LOADPIN 106
#define LSM_ID_SAFESETID 107
#define LSM_ID_LOCKDOWN 108
#define LSM_ID_BPF 109
#define LSM_ID_LANDLOCK 110
#define LSM_ID_IMA 111
#define LSM_ID_EVM 112
#define LSM_ID_IPE 113
#define LSM_ATTR_UNDEF 0
#define LSM_ATTR_CURRENT 100
#define LSM_ATTR_EXEC 101
#define LSM_ATTR_FSCREATE 102
#define LSM_ATTR_KEYCREATE 103
#define LSM_ATTR_PREV 104
#define LSM_ATTR_SOCKCREATE 105
#define LSM_FLAG_SINGLE 0x0001
#endif
