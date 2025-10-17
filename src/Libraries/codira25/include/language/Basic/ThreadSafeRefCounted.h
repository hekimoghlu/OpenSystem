/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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

//===--- ThreadSafeRefCounted.h - Thread-safe Refcounting Base --*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_THREADSAFEREFCOUNTED_H
#define LANGUAGE_BASIC_THREADSAFEREFCOUNTED_H

#include <atomic>
#include <cassert>
#include "toolchain/ADT/IntrusiveRefCntPtr.h"

namespace language {

/// A class that has the same function as \c ThreadSafeRefCountedBase, but with
/// a virtual destructor.
///
/// Should be used instead of \c ThreadSafeRefCountedBase for classes that
/// already have virtual methods to enforce dynamic allocation via 'new'.
/// FIXME: This should eventually move to toolchain.
class ThreadSafeRefCountedBaseVPTR {
  mutable std::atomic<unsigned> ref_cnt;
  virtual void anchor();

protected:
  ThreadSafeRefCountedBaseVPTR() : ref_cnt(0) {}
  virtual ~ThreadSafeRefCountedBaseVPTR() {}

public:
  void Retain() const {
    ref_cnt += 1;
  }

  void Release() const {
    int refCount = static_cast<int>(--ref_cnt);
    assert(refCount >= 0 && "Reference count was already zero.");
    if (refCount == 0) delete this;
  }
};

} // end namespace language

#endif // LANGUAGE_BASIC_THREADSAFEREFCOUNTED_H
