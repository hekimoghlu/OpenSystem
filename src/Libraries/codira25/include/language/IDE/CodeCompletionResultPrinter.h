/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

//===--- CodeCompletionResultPrinter.h --------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_IDE_CODECOMPLETIONRESULTPRINTER_H
#define LANGUAGE_IDE_CODECOMPLETIONRESULTPRINTER_H

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/raw_ostream.h"
#include "toolchain/Support/Allocator.h"

namespace language {

class NullTerminatedStringRef;

namespace ide {

class CodeCompletionResult;
class CodeCompletionString;

void printCodeCompletionResultDescription(const CodeCompletionResult &Result,
                                          toolchain::raw_ostream &OS,
                                          bool leadingPunctuation);

void printCodeCompletionResultDescriptionAnnotated(
    const CodeCompletionResult &Result, toolchain::raw_ostream &OS,
    bool leadingPunctuation);

void printCodeCompletionResultTypeName(
    const CodeCompletionResult &Result, toolchain::raw_ostream &OS);

void printCodeCompletionResultTypeNameAnnotated(
    const CodeCompletionResult &Result, toolchain::raw_ostream &OS);

void printCodeCompletionResultSourceText(
    const CodeCompletionResult &Result, toolchain::raw_ostream &OS);

/// Print 'FilterName' from \p str into memory managed by \p Allocator and
/// return it as \c NullTerminatedStringRef .
NullTerminatedStringRef
getCodeCompletionResultFilterName(const CodeCompletionString *Str,
                                  toolchain::BumpPtrAllocator &Allocator);

} // namespace ide
} // namespace language

#endif
