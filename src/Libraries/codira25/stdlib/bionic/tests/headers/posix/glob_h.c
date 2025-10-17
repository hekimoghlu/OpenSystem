/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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

#include <glob.h>

#include "header_checks.h"

static void glob_h() {
  TYPE(glob_t);
  STRUCT_MEMBER(glob_t, size_t, gl_pathc);
  STRUCT_MEMBER(glob_t, char**, gl_pathv);
  STRUCT_MEMBER(glob_t, size_t, gl_offs);
  TYPE(size_t);

  MACRO(GLOB_APPEND);
  MACRO(GLOB_DOOFFS);
  MACRO(GLOB_ERR);
  MACRO(GLOB_MARK);
  MACRO(GLOB_NOCHECK);
  MACRO(GLOB_NOESCAPE);
  MACRO(GLOB_NOSORT);

  MACRO(GLOB_ABORTED);
  MACRO(GLOB_NOMATCH);
  MACRO(GLOB_NOSPACE);

  FUNCTION(glob, int (*f)(const char*, int, int (*)(const char*, int), glob_t*));
  FUNCTION(globfree, void (*f)(glob_t*));
}
