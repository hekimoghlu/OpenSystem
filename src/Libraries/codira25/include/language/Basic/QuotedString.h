/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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

//===--- QuotedString.h - Print a string in double-quotes -------*- C++ -*-===//
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
/// \file Declares QuotedString, a convenient type for printing a
/// string as a string literal.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_QUOTEDSTRING_H
#define LANGUAGE_BASIC_QUOTEDSTRING_H

#include "toolchain/ADT/StringRef.h"

namespace toolchain {
  class raw_ostream;
}

namespace language {
  /// Print the given string as if it were a quoted string.
  void printAsQuotedString(toolchain::raw_ostream &out, toolchain::StringRef text);

  /// A class designed to make it easy to write a string to a stream
  /// as a quoted string.
  class QuotedString {
    toolchain::StringRef Text;
  public:
    explicit QuotedString(toolchain::StringRef text) : Text(text) {}

    friend toolchain::raw_ostream &operator<<(toolchain::raw_ostream &out,
                                         QuotedString string) {
      printAsQuotedString(out, string.Text);
      return out;
    }
  };
} // end namespace language

#endif // LANGUAGE_BASIC_QUOTEDSTRING_H
