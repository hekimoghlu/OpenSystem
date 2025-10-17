/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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

//===-- llvm/Support/Threading.h - Control multithreading mode --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TThis file defines llvm_start_multithreaded() and friends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_THREADING_H
#define LLVM_SYSTEM_THREADING_H

namespace llvm {
  /// llvm_start_multithreaded - Allocate and initialize structures needed to
  /// make LLVM safe for multithreading.  The return value indicates whether
  /// multithreaded initialization succeeded.  LLVM will still be operational
  /// on "failed" return, and will still be safe for hosting threading
  /// applications in the JIT, but will not be safe for concurrent calls to the
  /// LLVM APIs.
  /// THIS MUST EXECUTE IN ISOLATION FROM ALL OTHER LLVM API CALLS.
  bool llvm_start_multithreaded();

  /// llvm_stop_multithreaded - Deallocate structures necessary to make LLVM
  /// safe for multithreading.
  /// THIS MUST EXECUTE IN ISOLATION FROM ALL OTHER LLVM API CALLS.
  void llvm_stop_multithreaded();

  /// llvm_is_multithreaded - Check whether LLVM is executing in thread-safe
  /// mode or not.
  bool llvm_is_multithreaded();

  /// acquire_global_lock - Acquire the global lock.  This is a no-op if called
  /// before llvm_start_multithreaded().
  void llvm_acquire_global_lock();

  /// release_global_lock - Release the global lock.  This is a no-op if called
  /// before llvm_start_multithreaded().
  void llvm_release_global_lock();

  /// llvm_execute_on_thread - Execute the given \p UserFn on a separate
  /// thread, passing it the provided \p UserData.
  ///
  /// This function does not guarantee that the code will actually be executed
  /// on a separate thread or honoring the requested stack size, but tries to do
  /// so where system support is available.
  ///
  /// \param UserFn - The callback to execute.
  /// \param UserData - An argument to pass to the callback function.
  /// \param RequestedStackSize - If non-zero, a requested size (in bytes) for
  /// the thread stack.
  void llvm_execute_on_thread(void (*UserFn)(void*), void *UserData,
                              unsigned RequestedStackSize = 0);
}

#endif
