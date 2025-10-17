/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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

#include <grp.h>

#include "header_checks.h"

static void grp_h() {
  TYPE(struct group);
  STRUCT_MEMBER(struct group, char*, gr_name);
  STRUCT_MEMBER(struct group, gid_t, gr_gid);
  STRUCT_MEMBER(struct group, char**, gr_mem);

  TYPE(gid_t);
  TYPE(size_t);

  FUNCTION(endgrent, void (*f)(void));
  FUNCTION(getgrent, struct group* (*f)(void));
  FUNCTION(getgrgid, struct group* (*f)(gid_t));
  FUNCTION(getgrgid_r, int (*f)(gid_t, struct group*, char*, size_t, struct group**));
  FUNCTION(getgrnam, struct group* (*f)(const char*));
  FUNCTION(getgrnam_r, int (*f)(const char*, struct group*, char*, size_t, struct group**));
  FUNCTION(setgrent, void (*f)(void));
}
