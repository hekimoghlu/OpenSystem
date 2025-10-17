/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

//===--- IRGenRequests.cpp - Requests for LLVM IR Generation --------------===//
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

#include "language/AST/IRGenRequests.h"
#include "language/AST/ASTContext.h"
#include "language/AST/FileUnit.h"
#include "language/AST/Module.h"
#include "language/AST/SourceFile.h"
#include "language/SIL/SILModule.h"
#include "language/AST/TBDGenRequests.h"
#include "language/Subsystems.h"
#include "toolchain/IR/Module.h"
#include "toolchain/ExecutionEngine/Orc/ThreadSafeModule.h"

using namespace language;

namespace language {
// Implement the IRGen type zone (zone 20).
#define LANGUAGE_TYPEID_ZONE IRGen
#define LANGUAGE_TYPEID_HEADER "language/AST/IRGenTypeIDZone.def"
#include "language/Basic/ImplementTypeIDZone.h"
#undef LANGUAGE_TYPEID_ZONE
#undef LANGUAGE_TYPEID_HEADER
} // end namespace language

toolchain::orc::ThreadSafeModule GeneratedModule::intoThreadSafeContext() && {
  return {std::move(Module), std::move(Context)};
}

void language::simple_display(toolchain::raw_ostream &out,
                           const IRGenDescriptor &desc) {
  auto *MD = desc.Ctx.dyn_cast<ModuleDecl *>();
  if (MD) {
    out << "IR Generation for module " << MD->getName();
  } else {
    auto *file = desc.Ctx.get<FileUnit *>();
    out << "IR Generation for file ";
    simple_display(out, file);
  }
}

SourceLoc language::extractNearestSourceLoc(const IRGenDescriptor &desc) {
  return SourceLoc();
}

TinyPtrVector<FileUnit *> IRGenDescriptor::getFilesToEmit() const {
  // If we've been asked to emit a specific set of symbols, we don't emit any
  // whole files.
  if (SymbolsToEmit)
    return {};

  // For a whole module, we emit IR for all files.
  if (auto *mod = Ctx.dyn_cast<ModuleDecl *>())
    return TinyPtrVector<FileUnit *>(mod->getFiles());

  // For a primary file, we emit IR for both it and potentially its
  // SynthesizedFileUnit.
  auto *primary = Ctx.get<FileUnit *>();
  TinyPtrVector<FileUnit *> files;
  files.push_back(primary);

  return files;
}

ModuleDecl *IRGenDescriptor::getParentModule() const {
  if (auto *file = Ctx.dyn_cast<FileUnit *>())
    return file->getParentModule();
  return Ctx.get<ModuleDecl *>();
}

TBDGenDescriptor IRGenDescriptor::getTBDGenDescriptor() const {
  if (auto *file = Ctx.dyn_cast<FileUnit *>()) {
    return TBDGenDescriptor::forFile(file, TBDOpts);
  } else {
    auto *M = Ctx.get<ModuleDecl *>();
    return TBDGenDescriptor::forModule(M, TBDOpts);
  }
}

std::vector<std::string> IRGenDescriptor::getLinkerDirectives() const {
  auto desc = getTBDGenDescriptor();
  desc.getOptions().LinkerDirectivesOnly = true;
  return getPublicSymbols(std::move(desc));
}

evaluator::DependencySource IRGenRequest::readDependencySource(
    const evaluator::DependencyRecorder &e) const {
  auto &desc = std::get<0>(getStorage());

  // We don't track dependencies in whole-module mode.
  if (desc.Ctx.is<ModuleDecl *>()) {
    return nullptr;
  }

  auto *primary = desc.Ctx.get<FileUnit *>();
  return dyn_cast<SourceFile>(primary);
}

// Define request evaluation functions for each of the IRGen requests.
static AbstractRequestFunction *irGenRequestFunctions[] = {
#define LANGUAGE_REQUEST(Zone, Name, Sig, Caching, LocOptions)                    \
  reinterpret_cast<AbstractRequestFunction *>(&Name::evaluateRequest),
#include "language/AST/IRGenTypeIDZone.def"
#undef LANGUAGE_REQUEST
};

void language::registerIRGenRequestFunctions(Evaluator &evaluator) {
  evaluator.registerRequestFunctions(Zone::IRGen, irGenRequestFunctions);
}
