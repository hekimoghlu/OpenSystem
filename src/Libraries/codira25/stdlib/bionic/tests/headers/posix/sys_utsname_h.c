/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

#include <sys/utsname.h>

#include "header_checks.h"

static void sys_utsname_h() {
  TYPE(struct utsname);
  STRUCT_MEMBER_ARRAY(struct utsname, char/*[]*/, sysname);
  STRUCT_MEMBER_ARRAY(struct utsname, char/*[]*/, nodename);
  STRUCT_MEMBER_ARRAY(struct utsname, char/*[]*/, release);
  STRUCT_MEMBER_ARRAY(struct utsname, char/*[]*/, version);
  STRUCT_MEMBER_ARRAY(struct utsname, char/*[]*/, machine);

  FUNCTION(uname, int (*f)(struct utsname*));
}
