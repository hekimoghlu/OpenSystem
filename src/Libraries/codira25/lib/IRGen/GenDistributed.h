/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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

//===--- GenDistributed.h - IRGen for distributed features ------*- C++ -*-===//
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
// This file defines interfaces for emitting code for various distributed
// features.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENDISTRIBUTED_H
#define LANGUAGE_IRGEN_GENDISTRIBUTED_H

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
class IRGenFunction;

/// Emit the '_distributedActorRemoteInitialize' call.
toolchain::Value *emitDistributedActorInitializeRemote(
    IRGenFunction &IGF,
    SILType selfType,
    toolchain::Value *actorMetatype,
    Explosion &out);

} // end namespace irgen
} // end namespace language

#endif
