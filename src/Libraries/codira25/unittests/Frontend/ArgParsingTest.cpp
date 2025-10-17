/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 26, 2025.
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

//===-- ArgParsingTest.cpp --------------------------------------*- C++ -*-===//
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

#include "ArgParsingTest.h"

using namespace language;

ArgParsingTest::ArgParsingTest() : diags(sourceMgr) {}

void ArgParsingTest::parseArgs(const Args &args) {
  std::vector<const char *> adjustedArgs;

  if (this->langMode) {
    adjustedArgs.reserve(args.size() + 2);
    adjustedArgs.push_back("-language-version");
    adjustedArgs.push_back(this->langMode->data());
  } else {
    adjustedArgs.reserve(args.size());
  }

  for (auto &arg : args) {
    adjustedArgs.push_back(arg.data());
  }

  this->invocation.parseArgs(adjustedArgs, this->diags);
}

LangOptions &ArgParsingTest::getLangOptions() {
  return this->invocation.getLangOptions();
}

void PrintTo(const Args &value, std::ostream *os) {
  *os << '"';

  if (!value.empty()) {
    const auto lastIdx = value.size() - 1;
    for (size_t idx = 0; idx != lastIdx; ++idx) {
      *os << value[idx] << ' ';
    }
    *os << value[lastIdx];
  }

  *os << '"';
}
