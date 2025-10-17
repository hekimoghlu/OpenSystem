/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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

//===--- PatternMatching.cpp ----------------------------------------------===//
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

#include <IndexStoreDB_Support/PatternMatching.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>

using namespace IndexStoreDB;

static bool matchesPatternSubstring(StringRef Input,
                                    StringRef Pattern,
                                    bool AnchorStart,
                                    bool AnchorEnd,
                                    bool IgnoreCase) {
  if (AnchorStart && AnchorEnd) {
    if (IgnoreCase)
      return Input.equals_lower(Pattern);
    else
      return Input.equals(Pattern);
  }
  if (AnchorStart) {
    if (IgnoreCase)
      return Input.startswith_lower(Pattern);
    else
      return Input.startswith(Pattern);
  }
  if (AnchorEnd) {
    if (IgnoreCase)
      return Input.endswith_lower(Pattern);
    else
      return Input.endswith(Pattern);
  }
  if (IgnoreCase) {
    size_t N = Pattern.size();
    if (N > Input.size())
      return false;

    for (size_t i = 0, e = Input.size() - N + 1; i != e; ++i)
      if (Input.substr(i, N).equals_lower(Pattern))
        return true;
    return false;
  } else {
    return Input.find(Pattern) != StringRef::npos;
  }
}

static char ascii_tolower(char x) {
  if (x >= 'A' && x <= 'Z')
    return x - 'A' + 'a';
  return x;
}

static bool matchesPatternSubsequence(StringRef Input,
                                      StringRef Pattern,
                                      bool AnchorStart,
                                      bool AnchorEnd,
                                      bool IgnoreCase) {
  if (Input.empty() || Pattern.empty())
    return false;

  auto equals = [&](char c1, char c2)->bool {
    if (IgnoreCase)
      return ascii_tolower(c1) == ascii_tolower(c2);
    else
      return c1 == c2;
  };

  if (Input.size() < Pattern.size())
    return false;

  if (AnchorStart) {
    if (!equals(Input[0], Pattern[0]))
      return false;
  }

  while (!Input.empty() && !Pattern.empty()) {
    if (equals(Input[0], Pattern[0])) {
      Pattern = Pattern.substr(1);
    }
    Input = Input.substr(1);
  }

  if (!Pattern.empty())
    return false;
  if (AnchorEnd && !Input.empty())
    return false;

  return true;
}

bool IndexStoreDB::matchesPattern(StringRef Input,
                               StringRef Pattern,
                               bool AnchorStart,
                               bool AnchorEnd,
                               bool Subsequence,
                               bool IgnoreCase) {
  if (Subsequence)
    return matchesPatternSubsequence(Input, Pattern, AnchorStart, AnchorEnd, IgnoreCase);
  else
    return matchesPatternSubstring(Input, Pattern, AnchorStart, AnchorEnd, IgnoreCase);
}
