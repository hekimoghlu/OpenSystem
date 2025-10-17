/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

#include <dlfcn.h>

#include "header_checks.h"

static void dlfcn_h() {
  MACRO(RTLD_LAZY);
  MACRO(RTLD_NOW);
  MACRO(RTLD_GLOBAL);
  MACRO(RTLD_LOCAL);

#if !defined(__GLIBC__)  // Our glibc is too old.
  TYPE(Dl_info);
  STRUCT_MEMBER(Dl_info, const char*, dli_fname);
  STRUCT_MEMBER(Dl_info, void*, dli_fbase);
  STRUCT_MEMBER(Dl_info, const char*, dli_sname);
  STRUCT_MEMBER(Dl_info, void*, dli_saddr);
#endif

#if !defined(__GLIBC__)  // Our glibc is too old.
  FUNCTION(dladdr, int (*f)(const void*, Dl_info*));
#endif
  FUNCTION(dlclose, int (*f)(void*));
  FUNCTION(dlerror, char* (*f)(void));
  FUNCTION(dlopen, void* (*f)(const char*, int));
  FUNCTION(dlsym, void* (*f)(void*, const char*));
}
