/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

//===--- Heap.cpp - Heap tests --------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "language/Runtime/Heap.h"

#include "gtest/gtest.h"

void shouldAlloc(size_t size, size_t alignMask) {
  void *ptr = language::language_slowAlloc(size, alignMask);
  EXPECT_NE(ptr, (void *)NULL)
    << "Allocation failed for size " << size << " and alignment mask "
    << alignMask << ".";
  language::language_slowDealloc(ptr, size, alignMask);
}

void shouldAlloc(size_t size) {
  shouldAlloc(size, 0);
  shouldAlloc(size, 1);
  shouldAlloc(size, 3);
  shouldAlloc(size, 7);
  shouldAlloc(size, 15);
  shouldAlloc(size, 31);
  shouldAlloc(size, 63);
  shouldAlloc(size, 4095);
}

TEST(HeapTest, slowAlloc) {
  shouldAlloc(1);
  shouldAlloc(8);
  shouldAlloc(32);
  shouldAlloc(1093);
}

void shouldAllocTyped(size_t size, size_t alignMask, language::MallocTypeId typeId) {
  void *ptr = language::language_slowAllocTyped(size, alignMask, typeId);
  EXPECT_NE(ptr, (void *)NULL)
    << "Typed allocation failed for size " << size << " and alignment mask "
    << alignMask << ".";
  language::language_slowDealloc(ptr, size, alignMask);
}

void shouldAllocTyped(size_t size, language::MallocTypeId typeId) {
  shouldAlloc(size, 0);
  shouldAlloc(size, 1);
  shouldAlloc(size, 3);
  shouldAlloc(size, 7);
  shouldAlloc(size, 15);
  shouldAlloc(size, 31);
  shouldAlloc(size, 63);
  shouldAlloc(size, 4095);
}

void shouldAllocTyped(size_t size) {
  shouldAllocTyped(size, 42);
}

TEST(HeapTest, slowAllocTyped) {
  shouldAllocTyped(1);
  shouldAllocTyped(8);
  shouldAllocTyped(32);
  shouldAllocTyped(1093);
}

