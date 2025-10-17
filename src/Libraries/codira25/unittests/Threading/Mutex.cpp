/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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

//===--- Mutex.cpp - Mutex Tests ------------------------------------------===//
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

#include "language/Threading/Mutex.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <map>
#include <random>

#include "LockingHelpers.h"

using namespace language;

// -----------------------------------------------------------------------------

TEST(MutexTest, BasicLockable) {
  Mutex mutex(/* checked = */ true);
  basicLockable(mutex);
}

TEST(LazyMutexTest, BasicLockable) {
  static LazyMutex mutex;
  basicLockable(mutex);
}

TEST(LazyUnsafeMutexTest, BasicLockable) {
  static LazyUnsafeMutex mutex;
  basicLockable(mutex);
}

TEST(SmallMutex, BasicLockable) {
  SmallMutex mutex;
  basicLockable(mutex);
}

TEST(MutexTest, TryLockable) {
  Mutex mutex(/* checked = */ true);
  tryLockable(mutex);
}

TEST(LazyMutexTest, TryLockable) {
  static LazyMutex mutex;
  tryLockable(mutex);
}

TEST(LazyUnsafeMutexTest, TryLockable) {
  static LazyUnsafeMutex mutex;
  tryLockable(mutex);
}

TEST(SmallMutex, TryLockable) {
  SmallMutex mutex;
  tryLockable(mutex);
}

TEST(MutexTest, BasicLockableThreaded) {
  Mutex mutex(/* checked = */ true);
  basicLockableThreaded(mutex);
}

TEST(LazyMutexTest, BasicLockableThreaded) {
  static LazyMutex mutex;
  basicLockableThreaded(mutex);
}

TEST(LazyUnsafeMutexTest, BasicLockableThreaded) {
  static LazyUnsafeMutex mutex;
  basicLockableThreaded(mutex);
}

TEST(SmallMutex, BasicLockableThreaded) {
  SmallMutex mutex;
  basicLockableThreaded(mutex);
}

TEST(MutexTest, LockableThreaded) {
  Mutex mutex(/* checked = */ true);
  lockableThreaded(mutex);
}

TEST(LazyMutexTest, LockableThreaded) {
  static LazyMutex Mutex;
  lockableThreaded(Mutex);
}

TEST(SmallMutexTest, LockableThreaded) {
  SmallMutex Mutex;
  lockableThreaded(Mutex);
}

TEST(MutexTest, ScopedLockThreaded) {
  Mutex mutex(/* checked = */ true);
  scopedLockThreaded<Mutex::ScopedLock>(mutex);
}

TEST(LazyMutexTest, ScopedLockThreaded) {
  static LazyMutex Mutex;
  scopedLockThreaded<LazyMutex::ScopedLock>(Mutex);
}

TEST(SmallMutexTest, ScopedLockThreaded) {
  SmallMutex mutex(/* checked = */ true);
  scopedLockThreaded<ScopedLockT<SmallMutex, false>>(mutex);
}

TEST(MutexTest, ScopedUnlockUnderScopedLockThreaded) {
  Mutex mutex(/* checked = */ true);
  scopedUnlockUnderScopedLockThreaded<Mutex::ScopedLock, Mutex::ScopedUnlock>(
      mutex);
}

TEST(LazyMutexTest, ScopedUnlockUnderScopedLockThreaded) {
  static LazyMutex Mutex;
  scopedUnlockUnderScopedLockThreaded<LazyMutex::ScopedLock,
                                      LazyMutex::ScopedUnlock>(Mutex);
}

TEST(SmallMutexTest, ScopedUnlockUnderScopedLockThreaded) {
  SmallMutex mutex(/* checked = */ true);
  scopedUnlockUnderScopedLockThreaded<SmallMutex::ScopedLock,
                                      SmallMutex::ScopedUnlock>(mutex);
}

TEST(MutexTest, CriticalSectionThreaded) {
  Mutex mutex(/* checked = */ true);
  criticalSectionThreaded(mutex);
}

TEST(LazyMutexTest, CriticalSectionThreaded) {
  static LazyMutex Mutex;
  criticalSectionThreaded(Mutex);
}
