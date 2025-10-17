/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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

//===--- Overrides.h - Compat overrides for Codira 5.6 runtime ------s------===//
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
//  This file provides compatibility override hooks for Codira 5.6 runtimes.
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/Metadata.h"
#include "toolchain/ADT/StringRef.h"
#include "CompatibilityOverride.h"

namespace language {
struct OpaqueValue;
class AsyncContext;
class AsyncTask;

using TaskCreateCommon_t = LANGUAGE_CC(language) AsyncTaskAndContext(
    size_t rawTaskCreateFlags,
    TaskOptionRecord *options,
    const Metadata *futureResultType,
    TaskContinuationFunction *function, void *closureContext,
    size_t initialContextSize);

using TaskFutureWait_t = LANGUAGE_CC(languageasync) void(
                              OpaqueValue *result,
                              LANGUAGE_ASYNC_CONTEXT AsyncContext *callerContext,
                              AsyncTask *task,
                              TaskContinuationFunction *resumeFn,
                              AsyncContext *callContext);

using TaskFutureWaitThrowing_t = LANGUAGE_CC(languageasync) void(
                              OpaqueValue *result,
                              LANGUAGE_ASYNC_CONTEXT AsyncContext *callerContext,
                              AsyncTask *task,
                              ThrowingTaskFutureWaitContinuationFunction *resumeFn,
                              AsyncContext *callContext);

__attribute__((weak, visibility("hidden")))
void LANGUAGE_CC(languageasync) language56override_language_task_future_wait(
                                            OpaqueValue *,
                                            LANGUAGE_ASYNC_CONTEXT AsyncContext *,
                                            AsyncTask *,
                                            TaskContinuationFunction *,
                                            AsyncContext *,
                                            TaskFutureWait_t *original);

__attribute__((weak, visibility("hidden")))
void LANGUAGE_CC(languageasync) language56override_language_task_future_wait_throwing(
                                            OpaqueValue *,
                                            LANGUAGE_ASYNC_CONTEXT AsyncContext *,
                                            AsyncTask *,
                                            ThrowingTaskFutureWaitContinuationFunction *,
                                            AsyncContext *,
                                            TaskFutureWaitThrowing_t *original);

#if __POINTER_WIDTH__ == 64
__attribute__((weak, visibility("hidden")))
AsyncTaskAndContext LANGUAGE_CC(language)
language56override_language_task_create_common(
    size_t rawTaskCreateFlags,
    TaskOptionRecord *options,
    const Metadata *futureResultType,
    TaskContinuationFunction *function, void *closureContext,
    size_t initialContextSize,
    TaskCreateCommon_t *original);
#endif

} // namespace language
