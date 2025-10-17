/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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

#include <cpio.h>

#include "header_checks.h"

static void cpio_h() {
  MACRO_VALUE(C_IRUSR, 0400);
  MACRO_VALUE(C_IWUSR, 0200);
  MACRO_VALUE(C_IXUSR, 0100);

  MACRO_VALUE(C_IRGRP, 040);
  MACRO_VALUE(C_IWGRP, 020);
  MACRO_VALUE(C_IXGRP, 010);

  MACRO_VALUE(C_IROTH, 04);
  MACRO_VALUE(C_IWOTH, 02);
  MACRO_VALUE(C_IXOTH, 01);

  MACRO_VALUE(C_ISUID, 04000);
  MACRO_VALUE(C_ISGID, 02000);
  MACRO_VALUE(C_ISVTX, 01000);

  MACRO_VALUE(C_ISDIR, 040000);
  MACRO_VALUE(C_ISFIFO, 010000);
  MACRO_VALUE(C_ISREG, 0100000);
  MACRO_VALUE(C_ISBLK, 060000);
  MACRO_VALUE(C_ISCHR, 020000);

  MACRO_VALUE(C_ISCTG, 0110000);
  MACRO_VALUE(C_ISLNK, 0120000);
  MACRO_VALUE(C_ISSOCK, 0140000);

#if !defined(MAGIC)
#error MAGIC
#endif
}
