/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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

//===--- FileSystem.h - File helpers that interact with Diags ---*- C++ -*-===//
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

#ifndef LANGUAGE_AST_FILESYSTEM_H
#define LANGUAGE_AST_FILESYSTEM_H

#include "language/Basic/FileSystem.h"
#include "language/AST/DiagnosticEngine.h"
#include "language/AST/DiagnosticsCommon.h"
#include "toolchain/Support/VirtualOutputBackend.h"
#include "toolchain/Support/VirtualOutputConfig.h"

namespace language {
/// A wrapper around toolchain::vfs::OutputBackend to handle diagnosing any file
/// system errors during output creation.
///
/// \returns true if there were any errors, either from the filesystem
/// operations or from \p action returning true.
inline bool
withOutputPath(DiagnosticEngine &diags, toolchain::vfs::OutputBackend &Backend,
               StringRef outputPath,
               toolchain::function_ref<bool(toolchain::raw_pwrite_stream &)> action) {
  assert(!outputPath.empty());
  toolchain::vfs::OutputConfig config;
  config.setAtomicWrite().setOnlyIfDifferent();

  auto outputFile = Backend.createFile(outputPath, config);
  if (!outputFile) {
    diags.diagnose(SourceLoc(), diag::error_opening_output, outputPath,
                   toString(outputFile.takeError()));
    return true;
  }

  bool failed = action(*outputFile);
  // If there is an error, discard output. Otherwise keep the output file.
  if (auto error = failed ? outputFile->discard() : outputFile->keep()) {
    // Don't diagnose discard error.
    if (failed)
      consumeError(std::move(error));
    else
      diags.diagnose(SourceLoc(), diag::error_closing_output, outputPath,
                     toString(std::move(error)));
    return true;
  }
  return failed;
}
} // end namespace language

#endif // LANGUAGE_AST_FILESYSTEM_H
