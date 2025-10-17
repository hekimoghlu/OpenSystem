/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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

//===--- Refactoring.h - Framework for clang refactoring tools --*- C++ -*-===//
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
//
//  Interfaces supporting refactorings that span multiple translation units.
//  While single translation unit refactorings are supported via the Rewriter,
//  when refactoring multiple translation units changes must be stored in a
//  SourceManager independent form, duplicate changes need to be removed, and
//  all changes must be applied at once at the end of the refactoring so that
//  the code is always parseable.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_TOOLING_REFACTORING_H
#define LANGUAGE_CORE_TOOLING_REFACTORING_H

#include "language/Core/Tooling/Core/Replacement.h"
#include "language/Core/Tooling/Tooling.h"
#include <map>
#include <string>

namespace language::Core {

class Rewriter;

namespace tooling {

/// A tool to run refactorings.
///
/// This is a refactoring specific version of \see ClangTool. FrontendActions
/// passed to run() and runAndSave() should add replacements to
/// getReplacements().
class RefactoringTool : public ClangTool {
public:
  /// \see ClangTool::ClangTool.
  RefactoringTool(const CompilationDatabase &Compilations,
                  ArrayRef<std::string> SourcePaths,
                  std::shared_ptr<PCHContainerOperations> PCHContainerOps =
                      std::make_shared<PCHContainerOperations>());

  /// Returns the file path to replacements map to which replacements
  /// should be added during the run of the tool.
  std::map<std::string, Replacements> &getReplacements();

  /// Call run(), apply all generated replacements, and immediately save
  /// the results to disk.
  ///
  /// \returns 0 upon success. Non-zero upon failure.
  int runAndSave(FrontendActionFactory *ActionFactory);

  /// Apply all stored replacements to the given Rewriter.
  ///
  /// FileToReplaces will be deduplicated with `groupReplacementsByFile` before
  /// application.
  ///
  /// Replacement applications happen independently of the success of other
  /// applications.
  ///
  /// \returns true if all replacements apply. false otherwise.
  bool applyAllReplacements(Rewriter &Rewrite);

private:
  /// Write all refactored files to disk.
  int saveRewrittenFiles(Rewriter &Rewrite);

private:
  std::map<std::string, Replacements> FileToReplaces;
};

/// Groups \p Replaces by the file path and applies each group of
/// Replacements on the related file in \p Rewriter. In addition to applying
/// given Replacements, this function also formats the changed code.
///
/// \pre Replacements must be conflict-free.
///
/// FileToReplaces will be deduplicated with `groupReplacementsByFile` before
/// application.
///
/// Replacement applications happen independently of the success of other
/// applications.
///
/// \param[in] FileToReplaces Replacements (grouped by files) to apply.
/// \param[in] Rewrite The `Rewritter` to apply replacements on.
/// \param[in] Style The style name used for reformatting. See ```getStyle``` in
/// "include/clang/Format/Format.h" for all possible style forms.
///
/// \returns true if all replacements applied and formatted. false otherwise.
bool formatAndApplyAllReplacements(
    const std::map<std::string, Replacements> &FileToReplaces,
    Rewriter &Rewrite, StringRef Style = "file");

} // end namespace tooling
} // end namespace language::Core

#endif // LANGUAGE_CORE_TOOLING_REFACTORING_H
