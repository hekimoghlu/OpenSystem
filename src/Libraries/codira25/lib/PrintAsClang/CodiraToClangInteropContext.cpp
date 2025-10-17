/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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

//===--- CodiraToClangInteropContext.cpp - Interop context -------*- C++ -*-===//
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

#include "CodiraToClangInteropContext.h"
#include "language/AST/Decl.h"
#include "language/IRGen/IRABIDetailsProvider.h"

using namespace language;

CodiraToClangInteropContext::CodiraToClangInteropContext(
    ModuleDecl &mod, const IRGenOptions &irGenOpts)
    : mod(mod), irGenOpts(irGenOpts) {}

CodiraToClangInteropContext::~CodiraToClangInteropContext() {}

IRABIDetailsProvider &CodiraToClangInteropContext::getIrABIDetails() {
  if (!irABIDetails)
    irABIDetails = std::make_unique<IRABIDetailsProvider>(mod, irGenOpts);
  return *irABIDetails;
}

void CodiraToClangInteropContext::runIfStubForDeclNotEmitted(
    StringRef stubName, toolchain::function_ref<void(void)> function) {
  auto result = emittedStubs.insert(stubName);
  if (result.second)
    function();
}

void CodiraToClangInteropContext::recordExtensions(
    const NominalTypeDecl *typeDecl, const ExtensionDecl *ext) {
  auto it = extensions.insert(
      std::make_pair(typeDecl, std::vector<const ExtensionDecl *>()));
  it.first->second.push_back(ext);
}

toolchain::ArrayRef<const ExtensionDecl *>
CodiraToClangInteropContext::getExtensionsForNominalType(
    const NominalTypeDecl *typeDecl) const {
  auto exts = extensions.find(typeDecl);
  if (exts != extensions.end())
    return exts->getSecond();
  return {};
}
