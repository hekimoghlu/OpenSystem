/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

  MACRO(S_IFMT);
  MACRO(S_IFBLK);
  MACRO(S_IFCHR);
  MACRO(S_IFIFO);
  MACRO(S_IFREG);
  MACRO(S_IFDIR);
  MACRO(S_IFLNK);
  MACRO(S_IFSOCK);

  MACRO_VALUE(S_IRWXU, 0700);
  MACRO_VALUE(S_IRUSR, 0400);
  MACRO_VALUE(S_IWUSR, 0200);
  MACRO_VALUE(S_IXUSR, 0100);

  MACRO_VALUE(S_IRWXG, 070);
  MACRO_VALUE(S_IRGRP, 040);
  MACRO_VALUE(S_IWGRP, 020);
  MACRO_VALUE(S_IXGRP, 010);

  MACRO_VALUE(S_IRWXO, 07);
  MACRO_VALUE(S_IROTH, 04);
  MACRO_VALUE(S_IWOTH, 02);
  MACRO_VALUE(S_IXOTH, 01);

  MACRO_VALUE(S_ISUID, 04000);
  MACRO_VALUE(S_ISGID, 02000);
  MACRO_VALUE(S_ISVTX, 01000);
