/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

//===--- GenConcurrency.h - IRGen for concurrency features ------*- C++ -*-===//
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
// This file defines interfaces for emitting code for various concurrency
// features.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENCONCURRENCY_H
#define LANGUAGE_IRGEN_GENCONCURRENCY_H

#include "language/AST/Types.h"
#include "language/Basic/Toolchain.h"
#include "language/SIL/ApplySite.h"
#include "toolchain/IR/CallingConv.h"

#include "Callee.h"
#include "GenHeap.h"
#include "IRGenModule.h"

namespace toolchain {
class Value;
}

namespace language {
class CanType;
class ProtocolConformanceRef;
class SILType;

namespace irgen {
class Explosion;
class OptionalExplosion;
class IRGenFunction;

/// Emit the buildMainActorExecutorRef builtin.
void emitBuildMainActorExecutorRef(IRGenFunction &IGF, Explosion &out);

/// Emit the buildDefaultActorExecutorRef builtin.
void emitBuildDefaultActorExecutorRef(IRGenFunction &IGF, toolchain::Value *actor,
                                      Explosion &out);

/// Emit the buildOrdinaryTaskExecutorRef builtin.
void emitBuildOrdinaryTaskExecutorRef(
    IRGenFunction &IGF, toolchain::Value *executor, CanType executorType,
    ProtocolConformanceRef executorConformance, Explosion &out);

/// Emit the buildOrdinarySerialExecutorRef builtin.
void emitBuildOrdinarySerialExecutorRef(IRGenFunction &IGF,
                                        toolchain::Value *executor,
                                        CanType executorType,
                                        ProtocolConformanceRef executorConformance,
                                        Explosion &out);

/// Emit the buildComplexEqualitySerialExecutorRef builtin.
void emitBuildComplexEqualitySerialExecutorRef(IRGenFunction &IGF,
                                        toolchain::Value *executor,
                                        CanType executorType,
                                        ProtocolConformanceRef executorConformance,
                                        Explosion &out);

/// Emit the getCurrentExecutor builtin.
void emitGetCurrentExecutor(IRGenFunction &IGF, Explosion &out);

/// Emit the createAsyncLet builtin.
toolchain::Value *emitBuiltinStartAsyncLet(IRGenFunction &IGF,
                                      toolchain::Value *taskOptions,
                                      toolchain::Value *taskFunction,
                                      toolchain::Value *localContextInfo,
                                      toolchain::Value *resultBuffer,
                                      SubstitutionMap subs);

/// Emit the endAsyncLet builtin.
void emitEndAsyncLet(IRGenFunction &IGF, toolchain::Value *alet);

/// Emit the createTaskGroup builtin.
toolchain::Value *emitCreateTaskGroup(IRGenFunction &IGF, SubstitutionMap subs,
                                 toolchain::Value *groupFlags);

/// Emit the destroyTaskGroup builtin.
void emitDestroyTaskGroup(IRGenFunction &IGF, toolchain::Value *group);

void emitTaskRunInline(IRGenFunction &IGF, SubstitutionMap subs,
                       toolchain::Value *result, toolchain::Value *closure,
                       toolchain::Value *closureContext);

void emitTaskCancel(IRGenFunction &IGF, toolchain::Value *task);

toolchain::Value *maybeAddEmbeddedCodiraResultTypeInfo(IRGenFunction &IGF,
                                                 toolchain::Value *taskOptions,
                                                 CanType formalResultType);

/// Emit a call to language_task_create[_f] with the given flags, options, and
/// task function.
std::pair<toolchain::Value *, toolchain::Value *>
emitTaskCreate(IRGenFunction &IGF, toolchain::Value *flags,
               OptionalExplosion &initialExecutor,
               OptionalExplosion &taskGroup,
               OptionalExplosion &taskExecutorUnowned,
               OptionalExplosion &taskExecutorExistential,
               OptionalExplosion &taskName,
               Explosion &taskFunction,
               SubstitutionMap subs);

} // end namespace irgen
} // end namespace language

#endif
