/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#include <pwd.h>

#include "header_checks.h"

static void pwd_h() {
  TYPE(struct passwd);
  STRUCT_MEMBER(struct passwd, char*, pw_name);
  STRUCT_MEMBER(struct passwd, uid_t, pw_uid);
  STRUCT_MEMBER(struct passwd, gid_t, pw_gid);
  STRUCT_MEMBER(struct passwd, char*, pw_dir);
  STRUCT_MEMBER(struct passwd, char*, pw_shell);

  TYPE(gid_t);
  TYPE(uid_t);
  TYPE(size_t);

  FUNCTION(endpwent, void (*f)(void));
  FUNCTION(getpwent, struct passwd* (*f)(void));
  FUNCTION(getpwnam, struct passwd* (*f)(const char*));
  FUNCTION(getpwnam_r, int (*f)(const char*, struct passwd*, char*, size_t, struct passwd**));
  FUNCTION(getpwuid, struct passwd* (*f)(uid_t));
  FUNCTION(getpwuid_r, int (*f)(uid_t, struct passwd*, char*, size_t, struct passwd**));
  FUNCTION(setpwent, void (*f)(void));
}
