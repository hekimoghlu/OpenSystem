/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

//===--- PrettyStackTrace.cpp - Generic PrettyStackTraceEntries -----------===//
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
//  This file implements several PrettyStackTraceEntries that probably
//  ought to be in LLVM.
//
//===----------------------------------------------------------------------===//

#include "language/Basic/PrettyStackTrace.h"
#include "language/Basic/QuotedString.h"
#include "language/Basic/Version.h"
#include "toolchain/ADT/SmallString.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/raw_ostream.h"

using namespace language;

void PrettyStackTraceStringAction::print(toolchain::raw_ostream &out) const {
  out << "While " << Action << ' ' << QuotedString(TheString) << '\n';
}

void PrettyStackTraceFileContents::print(toolchain::raw_ostream &out) const {
  out << "Contents of " << Buffer.getBufferIdentifier() << ":\n---\n"
      << Buffer.getBuffer();
  if (!Buffer.getBuffer().ends_with("\n"))
    out << '\n';
  out << "---\n";
}

void PrettyStackTraceCodiraVersion::print(toolchain::raw_ostream &out) const {
  out << version::getCodiraFullVersion() << '\n';
}
