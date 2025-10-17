/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <sys/ipc.h>

#include "header_checks.h"

static void sys_ipc_h() {
  TYPE(struct ipc_perm);
  STRUCT_MEMBER(struct ipc_perm, uid_t, uid);
  STRUCT_MEMBER(struct ipc_perm, gid_t, gid);
  STRUCT_MEMBER(struct ipc_perm, uid_t, cuid);
  STRUCT_MEMBER(struct ipc_perm, gid_t, cgid);
#if defined(__GLIBC__)
  STRUCT_MEMBER(struct ipc_perm, unsigned short, mode);
#else
  STRUCT_MEMBER(struct ipc_perm, mode_t, mode);
#endif

  TYPE(uid_t);
  TYPE(gid_t);
  TYPE(mode_t);
  TYPE(key_t);

  MACRO(IPC_CREAT);
  MACRO(IPC_EXCL);
  MACRO(IPC_NOWAIT);

  MACRO(IPC_PRIVATE);

  MACRO(IPC_RMID);
  MACRO(IPC_SET);
  MACRO(IPC_STAT);

  FUNCTION(ftok, key_t (*f)(const char*, int));
}
