/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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

//===--- FixitApplyDiagnosticConsumer.h -------------------------*- C++ -*-===//
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
//
// This class records compiler interesting fix-its as textual edits.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_MIGRATOR_FIXITAPPLYDIAGNOSTICCONSUMER_H
#define LANGUAGE_MIGRATOR_FIXITAPPLYDIAGNOSTICCONSUMER_H

#include "language/AST/DiagnosticConsumer.h"
#include "language/Migrator/FixitFilter.h"
#include "language/Migrator/Migrator.h"
#include "language/Migrator/Replacement.h"
#include "language/Core/Rewrite/Core/RewriteBuffer.h"
#include "toolchain/ADT/SmallSet.h"

namespace language {

class CompilerInvocation;
struct DiagnosticInfo;
struct MigratorOptions;
class SourceManager;

namespace migrator {

struct Replacement;

class FixitApplyDiagnosticConsumer final
  : public DiagnosticConsumer, public FixitFilter {
  language::Core::RewriteBuffer RewriteBuf;

  /// The entire text of the input file.
  const StringRef Text;

  /// The name of the buffer, which should be the absolute path of the input
  /// filename.
  const StringRef BufferName;

  /// The number of fix-its pushed into the rewrite buffer. Use this to
  /// determine whether to call `printResult`.
  unsigned NumFixitsApplied;

  /// Tracks previous replacements so we don't pump the rewrite buffer with
  /// multiple equivalent replacements, which can result in weird behavior.
  toolchain::SmallSet<Replacement, 32> Replacements;

public:
  FixitApplyDiagnosticConsumer(const StringRef Text,
                               const StringRef BufferName);

  /// Print the resulting text, applying the caught fix-its to the given
  /// output stream.
  void printResult(toolchain::raw_ostream &OS) const;

  void handleDiagnostic(SourceManager &SM, const DiagnosticInfo &Info) override;

  unsigned getNumFixitsApplied() const {
    return NumFixitsApplied;
  }
};
} // end namespace migrator
} // end namespace language

#endif // LANGUAGE_MIGRATOR_FIXITAPPLYDIAGNOSTICCONSUMER_H
