/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
#pragma once

#include <android/dlext.h>
#include "linked_list.h"

#include <android-base/macros.h>

// Android uses RELA for LP64.
#if defined(__LP64__)
#define USE_RELA 1
#endif


struct soinfo;

class SoinfoListAllocator {
 public:
  static LinkedListEntry<soinfo>* alloc();
  static void free(LinkedListEntry<soinfo>* entry);

 private:
  // unconstructable
  DISALLOW_IMPLICIT_CONSTRUCTORS(SoinfoListAllocator);
};

class NamespaceListAllocator {
 public:
  static LinkedListEntry<android_namespace_t>* alloc();
  static void free(LinkedListEntry<android_namespace_t>* entry);

 private:
  // unconstructable
  DISALLOW_IMPLICIT_CONSTRUCTORS(NamespaceListAllocator);
};

typedef LinkedList<soinfo, SoinfoListAllocator> soinfo_list_t;
typedef LinkedList<android_namespace_t, NamespaceListAllocator> android_namespace_list_t;
