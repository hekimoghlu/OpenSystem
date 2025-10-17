/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

//===--- Concurrency.h - Runtime interface for concurrency ------*- C++ -*-===//
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
//
// The runtime interface for concurrency.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_RUNTIME_CONCURRENCY_BACKDEPLOY56_H
#define LANGUAGE_RUNTIME_CONCURRENCY_BACKDEPLOY56_H

#include "Concurrency/Task.h"
#include "Concurrency/TaskStatus.h"
#include "Concurrency/AsyncLet.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

// Does the runtime use a cooperative global executor?
#if defined(LANGUAGE_STDLIB_SINGLE_THREADED_RUNTIME)
#define LANGUAGE_CONCURRENCY_COOPERATIVE_GLOBAL_EXECUTOR 1
#else
#define LANGUAGE_CONCURRENCY_COOPERATIVE_GLOBAL_EXECUTOR 0
#endif

// Does the runtime integrate with libdispatch?
#ifndef LANGUAGE_CONCURRENCY_ENABLE_DISPATCH
#if LANGUAGE_CONCURRENCY_COOPERATIVE_GLOBAL_EXECUTOR
#define LANGUAGE_CONCURRENCY_ENABLE_DISPATCH 0
#else
#define LANGUAGE_CONCURRENCY_ENABLE_DISPATCH 1
#endif
#endif

namespace language {
class DefaultActor;
class TaskOptionRecord;

struct CodiraError;

struct AsyncTaskAndContext {
  AsyncTask *Task;
  AsyncContext *InitialContext;
};

/// Caution: not all future-initializing functions actually throw, so
/// this signature may be incorrect.
using FutureAsyncSignature =
  AsyncSignature<void(void*), /*throws*/ true>;

/// Escalate the priority of a task and all of its child tasks.
///
/// This can be called from any thread.
///
/// This has no effect if the task already has at least the given priority.
/// Returns the priority of the task.
LANGUAGE_CC(language)
__attribute__((visibility("hidden")))
JobPriority language_task_escalateBackdeploy56(AsyncTask *task,
                                            JobPriority newPriority);
} // namespace language

#pragma clang diagnostic pop

#endif // LANGUAGE_RUNTIME_CONCURRENCY_BACKDEPLOY56_H
