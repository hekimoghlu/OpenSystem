/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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

#include <sys/statvfs.h>

#include "header_checks.h"

static void sys_statvfs_h() {
  TYPE(struct statvfs);
  STRUCT_MEMBER(struct statvfs, unsigned long, f_bsize);
  STRUCT_MEMBER(struct statvfs, unsigned long, f_frsize);
  STRUCT_MEMBER(struct statvfs, fsblkcnt_t, f_blocks);
  STRUCT_MEMBER(struct statvfs, fsblkcnt_t, f_bfree);
  STRUCT_MEMBER(struct statvfs, fsblkcnt_t, f_bavail);
  STRUCT_MEMBER(struct statvfs, fsfilcnt_t, f_files);
  STRUCT_MEMBER(struct statvfs, fsfilcnt_t, f_ffree);
  STRUCT_MEMBER(struct statvfs, fsfilcnt_t, f_favail);
  STRUCT_MEMBER(struct statvfs, unsigned long, f_fsid);
  STRUCT_MEMBER(struct statvfs, unsigned long, f_flag);
  STRUCT_MEMBER(struct statvfs, unsigned long, f_namemax);

  TYPE(fsblkcnt_t);
  TYPE(fsfilcnt_t);

  MACRO(ST_RDONLY);
  MACRO(ST_NOSUID);

  FUNCTION(fstatvfs, int (*f)(int, struct statvfs*));
  FUNCTION(statvfs, int (*f)(const char*, struct statvfs*));
}
