/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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

#include <search.h>

#include "header_checks.h"

static void search_h() {
  TYPE(ENTRY);
  TYPE(struct entry);
  STRUCT_MEMBER(ENTRY, char*, key);
  STRUCT_MEMBER(ENTRY, void*, data);

  ACTION a;
  a = FIND;
  a = ENTER;

  VISIT v;
  v = preorder;
  v = postorder;
  v = endorder;
  v = leaf;

  TYPE(size_t);

  FUNCTION(hcreate, int (*f)(size_t));
  FUNCTION(hdestroy, void (*f)(void));
  FUNCTION(hsearch, ENTRY* (*f)(ENTRY, ACTION));
  FUNCTION(insque, void (*f)(void*, void*));
  FUNCTION(lfind, void* (*f)(const void*, const void*, size_t*, size_t,
                             int (*)(const void*, const void*)));
  FUNCTION(lsearch, void* (*f)(const void*, void*, size_t*, size_t,
                               int (*)(const void*, const void*)));
  FUNCTION(remque, void (*f)(void*));
  FUNCTION(tdelete, void* (*f)(const void*, void**, int (*)(const void*, const void*)));
  FUNCTION(tfind, void* (*f)(const void*, void* const*, int (*)(const void*, const void*)));
  FUNCTION(tsearch, void* (*f)(const void*, void**, int (*)(const void*, const void*)));
  FUNCTION(twalk, void (*f)(const void*, void (*)(const void*, VISIT, int)));
}
