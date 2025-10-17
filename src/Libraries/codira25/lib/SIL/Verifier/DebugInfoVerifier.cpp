/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

//===--- DebugInfoVerifier.cpp --------------------------------------------===//
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
///
/// \file
///
/// Utility verifier code for validating debug info.
///
//===----------------------------------------------------------------------===//

#include "language/Basic/Assertions.h"
#include "language/SIL/SILDebugScope.h"
#include "language/SIL/SILInstruction.h"

using namespace language;

//===----------------------------------------------------------------------===//
//                   MARK: Verify SILInstruction Debug Info
//===----------------------------------------------------------------------===//

void SILInstruction::verifyDebugInfo() const {
  auto require = [&](bool reqt, StringRef message) {
    if (!reqt) {
      toolchain::errs() << message << "\n";
      assert(false && "invoking standard assertion failure");
    }
  };

  // Check the location kind.
  SILLocation loc = getLoc();
  SILLocation::LocationKind locKind = loc.getKind();
  SILInstructionKind instKind = getKind();

  // Regular locations are allowed on all instructions.
  if (locKind == SILLocation::RegularKind)
    return;

  if (locKind == SILLocation::ReturnKind ||
      locKind == SILLocation::ImplicitReturnKind)
    require(
        instKind == SILInstructionKind::BranchInst ||
            instKind == SILInstructionKind::ReturnInst ||
            instKind == SILInstructionKind::UnreachableInst,
        "return locations are only allowed on branch and return instructions");

  if (locKind == SILLocation::ArtificialUnreachableKind)
    require(
        instKind == SILInstructionKind::UnreachableInst,
        "artificial locations are only allowed on Unreachable instructions");
}
