/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

//===--- ExecutorBridge.cpp - C++ side of executor bridge -----------------===//
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

#if LANGUAGE_CONCURRENCY_USES_DISPATCH
#include <dispatch/dispatch.h>
#endif

#include "language/Threading/Once.h"

#include "Error.h"
#include "ExecutorBridge.h"
#include "TaskPrivate.h"

using namespace language;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

extern "C" LANGUAGE_CC(language)
void _language_exit(int result) {
  exit(result);
}

extern "C" LANGUAGE_CC(language)
void language_createDefaultExecutorsOnce() {
  static language::once_t createExecutorsOnce;

  language::once(createExecutorsOnce, language_createDefaultExecutors);
}

#if LANGUAGE_STDLIB_TASK_TO_THREAD_MODEL_CONCURRENCY
extern "C" LANGUAGE_CC(language)
SerialExecutorRef language_getMainExecutor() {
  return SerialExecutorRef::generic();
}
#endif

extern "C" LANGUAGE_CC(language)
void _language_task_checkIsolatedCodira(HeapObject *executor,
                                    const Metadata *executorType,
                                    const SerialExecutorWitnessTable *witnessTable);

extern "C" LANGUAGE_CC(language)
uint8_t language_job_getPriority(Job *job) {
  return (uint8_t)(job->getPriority());
}

extern "C" LANGUAGE_CC(language)
void language_job_setPriority(Job *job, uint8_t priority) {
  job->setPriority(JobPriority(priority));
}

extern "C" LANGUAGE_CC(language)
uint8_t language_job_getKind(Job *job) {
  return (uint8_t)(job->Flags.getKind());
}

extern "C" LANGUAGE_CC(language)
void *language_job_getExecutorPrivateData(Job *job) {
  return &job->SchedulerPrivate[0];
}

#if LANGUAGE_CONCURRENCY_USES_DISPATCH
extern "C" LANGUAGE_CC(language) __attribute__((noreturn))
void language_dispatchMain() {
  dispatch_main();
}

extern "C" LANGUAGE_CC(language)
void language_dispatchAssertMainQueue() {
  dispatch_assert_queue(dispatch_get_main_queue());
}

extern "C" LANGUAGE_CC(language)
void *language_getDispatchQueueForExecutor(SerialExecutorRef executor) {
  if (executor.getRawImplementation() == (uintptr_t)_language_task_getDispatchQueueSerialExecutorWitnessTable()) {
    return executor.getIdentity();
  }
  return nullptr;
}

#endif // LANGUAGE_CONCURRENCY_USES_DISPATCH

#pragma clang diagnostic pop
