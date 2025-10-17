/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

// TEST_ENV MallocProbGuard=1 MallocProbGuardMemoryBudgetInKB=10000 MallocProbGuardSampleRate=1

import ObjectiveC
import Dispatch

// Race to be the first to get the name of a bunch of different generic classes.
// This tests the thread safety of lazy name installation. The MallocProbGuard
// variables help to more deterministically catch use-after-frees from this
// race. rdar://130280263
class C<T, U> {
  func doit(depth: Int, names: inout [String]) {
    if depth <= 0 { return }

    DispatchQueue.concurrentPerform(iterations: 2, execute: { _ in
      class_getName(object_getClass(self))
    })

    names.append(String(cString: class_getName(object_getClass(self))))

    C<T, C<T, U>>().doit(depth: depth - 1, names: &names)
    C<C<T, U>, U>().doit(depth: depth - 1, names: &names)
  }
}

var names: [String] = []
C<Int, Int>().doit(depth: 10, names: &names)

for name in names {
  _ = objc_getClass(name)
}

print("OK:", #file.split(separator: "/").last!)
