/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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

//===- GuessTargetAndModeCompilationDatabase.cpp --------------------------===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#include "language/Core/Tooling/CompilationDatabase.h"
#include "language/Core/Tooling/Tooling.h"
#include <memory>

namespace language::Core {
namespace tooling {

namespace {
class TargetAndModeAdderDatabase : public CompilationDatabase {
public:
  TargetAndModeAdderDatabase(std::unique_ptr<CompilationDatabase> Base)
      : Base(std::move(Base)) {
    assert(this->Base != nullptr);
  }

  std::vector<std::string> getAllFiles() const override {
    return Base->getAllFiles();
  }

  std::vector<CompileCommand> getAllCompileCommands() const override {
    return addTargetAndMode(Base->getAllCompileCommands());
  }

  std::vector<CompileCommand>
  getCompileCommands(StringRef FilePath) const override {
    return addTargetAndMode(Base->getCompileCommands(FilePath));
  }

private:
  std::vector<CompileCommand>
  addTargetAndMode(std::vector<CompileCommand> Cmds) const {
    for (auto &Cmd : Cmds) {
      if (Cmd.CommandLine.empty())
        continue;
      addTargetAndModeForProgramName(Cmd.CommandLine, Cmd.CommandLine.front());
    }
    return Cmds;
  }
  std::unique_ptr<CompilationDatabase> Base;
};
} // namespace

std::unique_ptr<CompilationDatabase>
inferTargetAndDriverMode(std::unique_ptr<CompilationDatabase> Base) {
  return std::make_unique<TargetAndModeAdderDatabase>(std::move(Base));
}

} // namespace tooling
} // namespace language::Core
