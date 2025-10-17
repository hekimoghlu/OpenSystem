/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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

//===--- InitializeDistActorIdentity.h - dist actor ID init for SILGen ----===//
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

#include "Cleanup.h"

namespace language {

namespace Lowering {

/// A clean-up designed to emit an initialization of a distributed actor's
/// identity upon successful initialization of the actor's system.
struct InitializeDistActorIdentity : Cleanup {
private:
  ConstructorDecl *ctor;
  ManagedValue actorSelf;
  VarDecl *systemVar;
public:
  InitializeDistActorIdentity(ConstructorDecl *ctor, ManagedValue actorSelf);

  void emit(SILGenFunction &SGF, CleanupLocation loc,
            ForUnwind_t forUnwind) override;

  void dump(SILGenFunction &) const override;

  VarDecl* getSystemVar() const { return systemVar; }
};

} // end Lowering namespace
} // end language namespace
