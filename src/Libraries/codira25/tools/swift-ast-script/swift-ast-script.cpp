/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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

//===--- language-ast-script.cpp ---------------------------------------------===//
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
/// This utility is a command line tool that searches Codira code for
/// declarations matching the given requirements.
///
//===----------------------------------------------------------------------===//

#include "language/Frontend/Frontend.h"
#include "language/FrontendTool/FrontendTool.h"
#include "language/Basic/ToolchainInitializer.h"
#include "ASTScript.h"
#include "ASTScriptConfiguration.h"

using namespace language;
using namespace scripting;

namespace {

class Observer : public FrontendObserver {
  ArrayRef<const char *> Args;
  std::unique_ptr<ASTScriptConfiguration> Config;
  std::unique_ptr<ASTScript> Script;
  bool HadError = false;

public:
  Observer(ArrayRef<const char *> args) : Args(args) {}

  void configuredCompiler(CompilerInstance &instance) override {
    Config = ASTScriptConfiguration::parse(instance, Args);
    if (!Config) return flagError();

    Script = ASTScript::parse(*Config);
    if (!Script) return flagError();
  }

  void performedSemanticAnalysis(CompilerInstance &instance) override {
    if (Script) {
      if (Script->execute())
        return flagError();
    }
  }

  bool hadError() const {
    return HadError;
  }

private:
  void flagError() {
    HadError = true;
  }
};

}

// ISO C++ does not allow 'main' to be used by a program [-Wmain]
int main2(int argc, const char *argv[]) {
  PROGRAM_START(argc, argv);

  // Look for the first "--" in the arguments.
  auto argBegin = argv + 1;
  auto argEnd = argv + argc;
  auto dashDash = std::find(argBegin, argEnd, StringRef("--"));
  if (dashDash == argEnd) {
    toolchain::errs() << "error: missing '--' in arguments to separate "
                    "script configuration from compiler arguments\n"
                    "usage:\n"
                    "  language-grep <script_file> -- <compiler flags>\n";
    return 1;
  }

  Observer observer(toolchain::ArrayRef(argBegin, dashDash));

  // Set up the frontend arguments.
  unsigned numFrontendArgs = argEnd - (dashDash + 1);
  SmallVector<const char *, 8> frontendArgs;
  frontendArgs.reserve(numFrontendArgs + 1);
  frontendArgs.append(dashDash + 1, argEnd);
  frontendArgs.push_back("-typecheck");

  int frontendResult =
    performFrontend(frontendArgs, argv[0], (void*) &main2, &observer);

  return (observer.hadError() ? 1 : frontendResult);
}

int main(int argc, const char *argv[]) {
  return main2(argc, argv);
}
