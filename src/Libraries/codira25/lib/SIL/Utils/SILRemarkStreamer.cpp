/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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

//===--- SILRemarkStreamer.cpp --------------------------------------------===//
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

#include "language/SIL/SILRemarkStreamer.h"
#include "language/AST/DiagnosticsFrontend.h"
#include "language/Basic/Assertions.h"
#include "toolchain/IR/LLVMContext.h"

using namespace language;

SILRemarkStreamer::SILRemarkStreamer(
    std::unique_ptr<toolchain::remarks::RemarkStreamer> &&streamer,
    std::unique_ptr<toolchain::raw_fd_ostream> &&stream, const ASTContext &Ctx)
    : owner(Owner::SILModule), streamer(std::move(streamer)), context(nullptr),
      remarkStream(std::move(stream)), ctx(Ctx) { }

toolchain::remarks::RemarkStreamer &SILRemarkStreamer::getLLVMStreamer() {
  switch (owner) {
  case Owner::SILModule:
    return *streamer.get();
  case Owner::LLVM:
    return *context->getMainRemarkStreamer();
  }
  return *streamer.get();
}

const toolchain::remarks::RemarkStreamer &
SILRemarkStreamer::getLLVMStreamer() const {
  switch (owner) {
  case Owner::SILModule:
    return *streamer.get();
  case Owner::LLVM:
    return *context->getMainRemarkStreamer();
  }
  return *streamer.get();
}

void SILRemarkStreamer::intoLLVMContext(toolchain::LLVMContext &Ctx) & {
  assert(owner == Owner::SILModule);
  Ctx.setMainRemarkStreamer(std::move(streamer));
  context = &Ctx;
  owner = Owner::LLVM;
}

std::unique_ptr<SILRemarkStreamer>
SILRemarkStreamer::create(SILModule &silModule) {
  StringRef filename = silModule.getOptions().OptRecordFile;
  const auto format = silModule.getOptions().OptRecordFormat;
  if (filename.empty())
    return nullptr;

  auto &diagEngine = silModule.getASTContext().Diags;
  std::error_code errorCode;
  auto file = std::make_unique<toolchain::raw_fd_ostream>(filename, errorCode,
                                                     toolchain::sys::fs::OF_None);
  if (errorCode) {
    diagEngine.diagnose(SourceLoc(), diag::cannot_open_file, filename,
                        errorCode.message());
    return nullptr;
  }

  toolchain::Expected<std::unique_ptr<toolchain::remarks::RemarkSerializer>>
      remarkSerializerOrErr = toolchain::remarks::createRemarkSerializer(
          format, toolchain::remarks::SerializerMode::Separate, *file);
  if (toolchain::Error err = remarkSerializerOrErr.takeError()) {
    diagEngine.diagnose(SourceLoc(), diag::error_creating_remark_serializer,
                        toString(std::move(err)));
    return nullptr;
  }

  auto mainRS = std::make_unique<toolchain::remarks::RemarkStreamer>(
      std::move(*remarkSerializerOrErr), filename);

  const auto passes = silModule.getOptions().OptRecordPasses;
  if (!passes.empty()) {
    if (toolchain::Error err = mainRS->setFilter(passes)) {
      diagEngine.diagnose(SourceLoc(), diag::error_creating_remark_serializer,
                          toString(std::move(err)));
      return nullptr;
    }
  }

  // N.B. We should be able to use std::make_unique here, but I prefer correctly
  // encapsulating the constructor over elegance.
  // Besides, this isn't going to throw an exception.
  return std::unique_ptr<SILRemarkStreamer>(new SILRemarkStreamer(
      std::move(mainRS), std::move(file), silModule.getASTContext()));
}
