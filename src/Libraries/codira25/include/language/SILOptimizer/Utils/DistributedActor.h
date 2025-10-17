/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

//===---- DistributedActor.h - SIL utils for distributed actors -*- C++ -*-===//
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

#ifndef LANGUAGE_SILOPTIMIZER_UTILS_DISTRIBUTED_ACTOR_H
#define LANGUAGE_SILOPTIMIZER_UTILS_DISTRIBUTED_ACTOR_H

#include "language/AST/Decl.h"
#include "toolchain/ADT/ArrayRef.h"
#include <optional>
#include <utility>

namespace language {

class ASTContext;
class ConstructorDecl;
class ClassDecl;
class DeclName;
class SILBasicBlock;
class SILBuilder;
class SILArgument;
class SILFunction;
class SILLocation;
class SILType;
class SILValue;

/// Creates a reference to the distributed actor's \p actorSystem
/// stored property.
SILValue refDistributedActorSystem(SILBuilder &B,
                                   SILLocation loc,
                                   ClassDecl *actorDecl,
                                   SILValue actorInstance);

/// Emit a call to a witness of the DistributedActorSystem protocol.
///
/// \param methodName The name of the method on the DistributedActorSystem protocol.
/// \param base The base on which to invoke the method
/// \param actorType If non-empty, the type of the distributed actor that is
/// provided as one of the arguments.
/// \param args The arguments provided to the call, not including the base.
/// \param tryTargets For a call that can throw, the normal and error basic
/// blocks that the call will branch to.
void emitDistributedActorSystemWitnessCall(
    SILBuilder &B, SILLocation loc, DeclName methodName, SILValue base,
    SILType actorType, toolchain::ArrayRef<SILValue> args,
    std::optional<std::pair<SILBasicBlock *, SILBasicBlock *>> tryTargets =
        std::nullopt);

/// Emits code that notifies the distributed actor's actorSystem that the
/// actor is ready for execution.
/// \param B the builder to use when emitting the code.
/// \param actor the distributed actor instance to pass to the actorSystem as
/// being "ready" \param actorSystem a value representing the DistributedActorSystem
void emitActorReadyCall(SILBuilder &B, SILLocation loc, SILValue actor,
                        SILValue actorSystem);

/// Emits code to notify the \p actorSystem that the given identity is resigned.
void emitResignIdentityCall(SILBuilder &B, SILLocation loc, ClassDecl* actDecl,
                            SILValue actor, SILValue identityRef);


} // namespace language

#endif
