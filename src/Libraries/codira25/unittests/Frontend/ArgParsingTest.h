/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

//===-- ArgParsingTest.h ----------------------------------------*- C++ -*-===//
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

#ifndef ARG_PARSING_TEST_H
#define ARG_PARSING_TEST_H

#include "language/Frontend/Frontend.h"
#include "gtest/gtest.h"

using Args = std::vector<std::string>;

class ArgParsingTest : public ::testing::Test {
  language::CompilerInvocation invocation;
  language::SourceManager sourceMgr;
  language::DiagnosticEngine diags;

protected:
  std::optional<std::string> langMode;

public:
  ArgParsingTest();

  void parseArgs(const Args &args);

  language::LangOptions &getLangOptions();
};

template <typename T>
struct ArgParsingTestCase final {
  Args args;
  T expectedResult;

  ArgParsingTestCase(Args args, T expectedResult)
      : args(std::move(args)), expectedResult(std::move(expectedResult)) {}
};

// MARK: - Printers

void PrintTo(const Args &, std::ostream *);

template <typename T>
void PrintTo(const ArgParsingTestCase<T> &value, std::ostream *os) {
  PrintTo(value.args, os);
}

#endif // ARG_PARSING_TEST_H
