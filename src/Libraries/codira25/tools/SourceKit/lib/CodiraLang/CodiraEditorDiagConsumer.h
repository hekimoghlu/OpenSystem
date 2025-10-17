/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

//===--- CodiraEditorDiagConsumer.h - ----------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_SOURCEKIT_LIB_LANGUAGELANG_LANGUAGEEDITORDIAGCONSUMER_H
#define TOOLCHAIN_SOURCEKIT_LIB_LANGUAGELANG_LANGUAGEEDITORDIAGCONSUMER_H

#include "SourceKit/Core/LangSupport.h"
#include "language/AST/DiagnosticConsumer.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/StringMap.h"

namespace SourceKit {

class EditorDiagConsumer : public language::DiagnosticConsumer {
  typedef std::vector<DiagnosticEntryInfo> DiagnosticsTy;
  /// Maps from a BufferID to the diagnostics that were emitted inside that
  /// buffer.
  toolchain::DenseMap<unsigned, DiagnosticsTy> BufferDiagnostics;
  DiagnosticsTy InvalidLocDiagnostics;

  toolchain::StringMap<BufferInfoSharedPtr> BufferInfos;

  int LastDiagBufferID = -1;
  unsigned LastDiagIndex = 0;

  bool haveLastDiag() {
    return LastDiagBufferID >= 0;
  }
  void clearLastDiag() {
    LastDiagBufferID = -1;
  }
  DiagnosticEntryInfo &getLastDiag() {
    assert(haveLastDiag());
    return BufferDiagnostics[LastDiagBufferID][LastDiagIndex];
  }

  bool HadAnyError = false;

  BufferInfoSharedPtr getBufferInfo(StringRef FileName,
                                    std::optional<unsigned> BufferID,
                                    language::SourceManager &SM);

public:
  /// The diagnostics are returned in source-order.
  ArrayRef<DiagnosticEntryInfo> getDiagnosticsForBuffer(unsigned BufferID) const {
    ArrayRef<DiagnosticEntryInfo> Diags;
    auto DiagFound = BufferDiagnostics.find(BufferID);
    if (DiagFound != BufferDiagnostics.end())
      Diags = DiagFound->second;
    return Diags;
  }

  void getAllDiagnostics(SmallVectorImpl<DiagnosticEntryInfo> &Result);

  bool hadAnyError() const { return HadAnyError; }

  void handleDiagnostic(language::SourceManager &SM,
                        const language::DiagnosticInfo &Info) override;
};

} // namespace SourceKit

#endif
