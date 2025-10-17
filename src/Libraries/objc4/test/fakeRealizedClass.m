/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
#include "test.h"
#include "class-structures.h"

#include <objc/NSObject.h>

#define RW_REALIZED (1U<<31)

// This test only runs on macOS, so we won't bother with the conditionals around
// this value. Just use the one value macOS always has.
#define FAST_IS_RW_POINTER      0x8000000000000000UL

__attribute__((section("__DATA,__objc_const")))
struct ObjCClass_ro FakeSuperclassRO = {
    .flags = RW_REALIZED
};

struct ObjCClass FakeSuperclass = {
    &OBJC_METACLASS_$_NSObject,
    &OBJC_METACLASS_$_NSObject,
    NULL,
    0,
    (struct ObjCClass_ro *)((uintptr_t)&FakeSuperclassRO + FAST_IS_RW_POINTER)
};

__attribute__((section("__DATA,__objc_const")))
struct ObjCClass_ro FakeSubclassRO;

struct ObjCClass FakeSubclass = {
  &FakeSuperclass,
  &FakeSuperclass,
  NULL,
  0,
  &FakeSubclassRO
};

static struct ObjCClass *class_ptr __attribute__((used)) __attribute((section("__DATA,__objc_nlclslist"))) = &FakeSubclass;

int main() {}
