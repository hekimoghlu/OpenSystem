/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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

#include <dirent.h>

#include "header_checks.h"

static void dirent_h() {
  INCOMPLETE_TYPE(DIR);

  TYPE(struct dirent);
#if defined(__BIONIC__) && !defined(__LP64__) // Historical ABI accident.
  STRUCT_MEMBER(struct dirent, uint64_t, d_ino);
#else
  STRUCT_MEMBER(struct dirent, ino_t, d_ino);
#endif
  STRUCT_MEMBER_ARRAY(struct dirent, char/*[]*/, d_name);

  TYPE(ino_t);

  FUNCTION(alphasort, int (*f)(const struct dirent**, const struct dirent**));
  FUNCTION(closedir, int (*f)(DIR*));
  FUNCTION(dirfd, int (*f)(DIR*));
  FUNCTION(fdopendir, DIR* (*f)(int));
  FUNCTION(opendir, DIR* (*f)(const char*));
  FUNCTION(readdir, struct dirent* (*f)(DIR*));
  FUNCTION(readdir_r, int (*f)(DIR*, struct dirent*, struct dirent**));
  FUNCTION(rewinddir, void (*f)(DIR*));
  FUNCTION(scandir, int (*f)(const char*, struct dirent***,
                             int (*)(const struct dirent*),
                             int (*)(const struct dirent**, const struct dirent**)));
  FUNCTION(seekdir, void (*f)(DIR*, long));
  FUNCTION(telldir, long (*f)(DIR*));
}
