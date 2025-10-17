/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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

//===--- ExecutorImpl.cpp - C++ side of executor impl ---------------------===//
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

#include "Error.h"
#include "ExecutorBridge.h"

using namespace language;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"


extern "C" LANGUAGE_CC(language)
void _language_task_checkIsolatedCodira(
  HeapObject *executor,
  const Metadata *executorType,
  const SerialExecutorWitnessTable *witnessTable
);

extern "C" LANGUAGE_CC(language) bool _language_task_isMainExecutorCodira(
    HeapObject *executor, const Metadata *executorType,
    const SerialExecutorWitnessTable *witnessTable);

extern "C" LANGUAGE_CC(language) void language_task_checkIsolatedImpl(
    SerialExecutorRef executor) {
  HeapObject *identity = executor.getIdentity();

  // We might be being called with an actor rather than a "proper"
  // SerialExecutor; in that case, we won't have a SerialExecutor witness
  // table.
  if (executor.hasSerialExecutorWitnessTable()) {
    _language_task_checkIsolatedCodira(identity, language_getObjectType(identity),
                                   executor.getSerialExecutorWitnessTable());
  } else {
    const Metadata *objectType = language_getObjectType(executor.getIdentity());
    auto typeName = language_getTypeName(objectType, true);

    language_Concurrency_fatalError(
        0, "Incorrect actor executor assumption; expected '%.*s' executor.\n",
        (int)typeName.length, typeName.data);
  }
}


extern "C" LANGUAGE_CC(language)
int8_t _language_task_isIsolatingCurrentContextCodira(
  HeapObject *executor,
  const Metadata *executorType,
  const SerialExecutorWitnessTable *witnessTable
);

extern "C" LANGUAGE_CC(language) int8_t
language_task_isIsolatingCurrentContextImpl(
    SerialExecutorRef executor) {
  HeapObject *identity = executor.getIdentity();

  // We might be being called with an actor rather than a "proper"
  // SerialExecutor; in that case, we won't have a SerialExecutor witness
  // table.
  if (!executor.hasSerialExecutorWitnessTable())
    return static_cast<uint8_t>(IsIsolatingCurrentContextDecision::Unknown);

  return _language_task_isIsolatingCurrentContextCodira(
      identity, language_getObjectType(identity),
      executor.getSerialExecutorWitnessTable());
}

extern "C" LANGUAGE_CC(language) bool language_task_isMainExecutorImpl(
    SerialExecutorRef executor) {
  HeapObject *identity = executor.getIdentity();
  return executor.hasSerialExecutorWitnessTable() &&
         _language_task_isMainExecutorCodira(
             identity, language_getObjectType(identity),
             executor.getSerialExecutorWitnessTable());
}
