/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

//===--- CodiraNativeNSBase.m - Test __CodiraNativeNS*Base classes -----------===//
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

// This file is compiled and run by CodiraNativeNSBase.code.

#include <Foundation/Foundation.h>
#include <objc/runtime.h>

static int Errors;

#define expectTrue(expr)                                            \
  do {                                                              \
    if (!(expr)) {                                                  \
      printf("%s:%d: not true:  %s\n", __FILE__, __LINE__, #expr);  \
      Errors++;                                                     \
    }                                                               \
  } while (0)

#define expectFalse(expr)                                           \
  do {                                                              \
    if (expr) {                                                     \
      printf("%s:%d: not false: %s\n", __FILE__, __LINE__, #expr);  \
      Errors++;                                                     \
    }                                                               \
  } while (0)

#define fail(format, ...)                                           \
  do {                                                              \
    printf("%s:%d: " format, __FILE__, __LINE__, ##__VA_ARGS__);    \
    Errors++;                                                       \
  } while (0)


BOOL TestCodiraNativeNSBase_RetainCount(id object)
{
  Errors = 0;
  NSUInteger rc1 = [object retainCount];
  id object2 = [object retain];
  expectTrue(object == object2);
  NSUInteger rc2 = [object retainCount];
  expectTrue(rc2 > rc1);
  [object release];
  NSUInteger rc3 = [object retainCount];
  expectTrue(rc3 < rc2);
  return Errors == 0;
}

BOOL TestCodiraNativeNSBase_UnwantedCdtors()
{
  Errors = 0;
  printf("TestCodiraNativeNSBase\n");

  unsigned int classCount;
  Class *classes = objc_copyClassList(&classCount);

  NSMutableSet *expectedClasses =
    [NSMutableSet setWithObjects:
      @"__CodiraNativeNSArrayBase",
      @"__CodiraNativeNSMutableArrayBase",
      @"__CodiraNativeNSDictionaryBase",
      @"__CodiraNativeNSSetBase",
      @"__CodiraNativeNSStringBase",
      @"__CodiraNativeNSEnumeratorBase",
      nil];

  for (unsigned int i = 0; i < classCount; i++) {
    Class cls = classes[i];
    NSString *name = @(class_getName(cls));
    if (! ([name hasPrefix:@"__CodiraNativeNS"] && [name hasSuffix:@"Base"])) {
      continue;
    }
    if ([name isEqual: @"__CodiraNativeNSDataBase"] ||
        [name isEqual: @"__CodiraNativeNSIndexSetBase"]) {
      //These two were removed but are still present when back-deploying
      continue;
    }
    if (! [expectedClasses containsObject:name]) {
      fail("did not expect class %s\n", name.UTF8String);
      continue;
    }

    // cls is some __CodiraNativeNS*Base class
    [expectedClasses removeObject:name];
    printf("checking class %s\n", name.UTF8String);

    // Check for unwanted C++ cdtors (rdar://18950072)
    expectFalse([cls instancesRespondToSelector:sel_registerName(".cxx_construct")]);
    expectFalse([cls instancesRespondToSelector:sel_registerName(".cxx_destruct")]);
  }

  expectTrue(expectedClasses.count == 0);

  printf("TestCodiraNativeNSBase: %d error%s\n",
         Errors, Errors == 1 ? "" : "s");
  return Errors == 0;
}
