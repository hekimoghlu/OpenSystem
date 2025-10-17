/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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

//===--- PrettyStackTrace.h - Generic stack-trace prettifiers ---*- C++ -*-===//
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

#ifndef LANGUAGE_BASIC_PRETTYSTACKTRACE_H
#define LANGUAGE_BASIC_PRETTYSTACKTRACE_H

#include "toolchain/Support/PrettyStackTrace.h"
#include "toolchain/ADT/StringRef.h"

namespace toolchain {
  class MemoryBuffer;
}

namespace language {

/// A PrettyStackTraceEntry for performing an action involving a StringRef.
///
/// The message is:
///   While <action> "<string>"\n
class PrettyStackTraceStringAction : public toolchain::PrettyStackTraceEntry {
  const char *Action;
  toolchain::StringRef TheString;
public:
  PrettyStackTraceStringAction(const char *action, toolchain::StringRef string)
    : Action(action), TheString(string) {}
  void print(toolchain::raw_ostream &OS) const override;
};

/// A PrettyStackTraceEntry to dump the contents of a file.
class PrettyStackTraceFileContents : public toolchain::PrettyStackTraceEntry {
  const toolchain::MemoryBuffer &Buffer;
public:
  explicit PrettyStackTraceFileContents(const toolchain::MemoryBuffer &buffer)
    : Buffer(buffer) {}
  void print(toolchain::raw_ostream &OS) const override;
};

/// A PrettyStackTraceEntry to print the version of the compiler.
class PrettyStackTraceCodiraVersion : public toolchain::PrettyStackTraceEntry {
public:
  void print(toolchain::raw_ostream &OS) const override;
};

} // end namespace language

#endif // LANGUAGE_BASIC_PRETTYSTACKTRACE_H
