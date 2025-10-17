/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#include "class-structures.h"
#include "test.h"

#include <objc/objc-abi.h>

#define BAD_ROOT_ADDRESS 0xbad4007

static struct ObjCCategory testCategory = {
    "TestCategory",
    (struct ObjCClass *)BAD_ROOT_ADDRESS
};

static struct ObjCCategory *testCategoryListEntry __attribute__((used)) __attribute__((section("__DATA, __objc_catlist"))) = &testCategory;


int main() {
    fail("This test is supposed to crash before main()");
}
