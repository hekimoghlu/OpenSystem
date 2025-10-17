/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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

//===--- TokenAnalyzer.h - Analyze Token Streams ----------------*- C++ -*-===//
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
///
/// \file
/// This file declares an abstract TokenAnalyzer, and associated helper
/// classes. TokenAnalyzer can be extended to generate replacements based on
/// an annotated and pre-processed token stream.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_LIB_FORMAT_TOKENANALYZER_H
#define LANGUAGE_CORE_LIB_FORMAT_TOKENANALYZER_H

#include "AffectedRangeManager.h"
#include "FormatTokenLexer.h"
#include "TokenAnnotator.h"

namespace language::Core {
namespace format {

class Environment {
public:
  // This sets up an virtual file system with file \p FileName containing the
  // fragment \p Code. Assumes that \p Code starts at \p FirstStartColumn,
  // that the next lines of \p Code should start at \p NextStartColumn, and
  // that \p Code should end at \p LastStartColumn if it ends in newline.
  // See also the documentation of language::Core::format::internal::reformat.
  Environment(StringRef Code, StringRef FileName, unsigned FirstStartColumn = 0,
              unsigned NextStartColumn = 0, unsigned LastStartColumn = 0);

  FileID getFileID() const { return ID; }

  SourceManager &getSourceManager() const { return SM; }

  ArrayRef<CharSourceRange> getCharRanges() const { return CharRanges; }

  // Returns the column at which the fragment of code managed by this
  // environment starts.
  unsigned getFirstStartColumn() const { return FirstStartColumn; }

  // Returns the column at which subsequent lines of the fragment of code
  // managed by this environment should start.
  unsigned getNextStartColumn() const { return NextStartColumn; }

  // Returns the column at which the fragment of code managed by this
  // environment should end if it ends in a newline.
  unsigned getLastStartColumn() const { return LastStartColumn; }

  // Returns nullptr and prints a diagnostic to stderr if the environment
  // can't be created.
  static std::unique_ptr<Environment> make(StringRef Code, StringRef FileName,
                                           ArrayRef<tooling::Range> Ranges,
                                           unsigned FirstStartColumn = 0,
                                           unsigned NextStartColumn = 0,
                                           unsigned LastStartColumn = 0);

private:
  // This is only set if constructed from string.
  std::unique_ptr<SourceManagerForFile> VirtualSM;

  // This refers to either a SourceManager provided by users or VirtualSM
  // created for a single file.
  SourceManager &SM;
  FileID ID;

  SmallVector<CharSourceRange, 8> CharRanges;
  unsigned FirstStartColumn;
  unsigned NextStartColumn;
  unsigned LastStartColumn;
};

class TokenAnalyzer : public UnwrappedLineConsumer {
public:
  TokenAnalyzer(const Environment &Env, const FormatStyle &Style);

  std::pair<tooling::Replacements, unsigned>
  process(bool SkipAnnotation = false);

protected:
  virtual std::pair<tooling::Replacements, unsigned>
  analyze(TokenAnnotator &Annotator,
          SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
          FormatTokenLexer &Tokens) = 0;

  void consumeUnwrappedLine(const UnwrappedLine &TheLine) override;

  void finishRun() override;

  FormatStyle Style;
  LangOptions LangOpts;
  // Stores Style, FileID and SourceManager etc.
  const Environment &Env;
  // AffectedRangeMgr stores ranges to be fixed.
  AffectedRangeManager AffectedRangeMgr;
  SmallVector<SmallVector<UnwrappedLine, 16>, 2> UnwrappedLines;
  encoding::Encoding Encoding;
};

} // end namespace format
} // end namespace language::Core

#endif
