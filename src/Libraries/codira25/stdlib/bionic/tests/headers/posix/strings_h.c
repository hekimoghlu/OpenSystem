/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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

#include <strings.h>

#include "header_checks.h"

static void strings_h() {
  FUNCTION(ffs, int (*f)(int));
#if !defined(__GLIBC__)
  FUNCTION(ffsl, int (*f)(long));
  FUNCTION(ffsll, int (*f)(long long));
#endif
  FUNCTION(strcasecmp, int (*f)(const char*, const char*));
  FUNCTION(strcasecmp_l, int (*f)(const char*, const char*, locale_t));
  FUNCTION(strncasecmp, int (*f)(const char*, const char*, size_t));
  FUNCTION(strncasecmp_l, int (*f)(const char*, const char*, size_t, locale_t));

  TYPE(locale_t);
  TYPE(size_t);
}
