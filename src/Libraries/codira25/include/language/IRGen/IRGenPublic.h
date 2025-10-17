/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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

//===--- IRGenPublic.h - Public interface to IRGen --------------*- C++ -*-===//
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
#ifndef LANGUAGE_IRGEN_IRGENPUBLIC_H
#define LANGUAGE_IRGEN_IRGENPUBLIC_H

namespace toolchain {
  class LLVMContext;
  template<typename T, unsigned N> class SmallVector;
}

namespace language {
class ASTContext;
class LinkLibrary;
class SILModule;

namespace irgen {

class IRGenerator;
class IRGenModule;

/// Create an IRGen module.
std::pair<IRGenerator *, IRGenModule *>
createIRGenModule(SILModule *SILMod, StringRef OutputFilename,
                  StringRef MainInputFilenameForDebugInfo,
                  StringRef PrivateDiscriminator, IRGenOptions &options);

/// Delete the IRGenModule and IRGenerator obtained by the above call.
void deleteIRGenModule(std::pair<IRGenerator *, IRGenModule *> &Module);

/// Gets the ABI version number that'll be set as a flag in the module.
uint32_t getCodiraABIVersion();

} // end namespace irgen
} // end namespace language

#endif
