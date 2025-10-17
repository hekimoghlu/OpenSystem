/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#include <ftw.h>

#include "header_checks.h"

static void ftw_h() {
  TYPE(struct FTW);
  STRUCT_MEMBER(struct FTW, int, base);
  STRUCT_MEMBER(struct FTW, int, level);

  MACRO(FTW_F);
  MACRO(FTW_D);
  MACRO(FTW_DNR);
  MACRO(FTW_DP);
  MACRO(FTW_NS);
  MACRO(FTW_SL);
  MACRO(FTW_SLN);

  MACRO(FTW_PHYS);
  MACRO(FTW_MOUNT);
  MACRO(FTW_DEPTH);
  MACRO(FTW_CHDIR);

  FUNCTION(ftw, int (*f)(const char*, int (*)(const char*, const struct stat*, int), int));

  TYPE(struct stat);

  // POSIX: "The <ftw.h> header shall define the ... the symbolic names for
  // st_mode and the file type test macros as described in <sys/stat.h>."
#include "sys_stat_h_mode_constants.h"
#include "sys_stat_h_file_type_test_macros.h"
}
