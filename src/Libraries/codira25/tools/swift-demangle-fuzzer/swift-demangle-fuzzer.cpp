/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

//===--- language-demangle-fuzzer.cpp - Codira fuzzer -------------------------===//
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
// This program tries to fuzz the demangler shipped as part of the language
// compiler.
// For this to work you need to pass --enable-sanitizer-coverage to build-script
// otherwise the fuzzer doesn't have coverage information to make progress
// (making the whole fuzzing operation really ineffective).
// It is recommended to use the tool together with another sanitizer to expose
// more bugs (asan, lsan, etc...)
//
//===----------------------------------------------------------------------===//

#include "language/Demangling/Demangle.h"
#include "language/Demangling/ManglingMacros.h"
#include <stddef.h>
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  language::Demangle::Context DCtx;
  language::Demangle::DemangleOptions Opts;
  std::string NullTermStr((const char *)Data, Size);
  language::Demangle::NodePointer pointer = DCtx.demangleSymbolAsNode(NullTermStr);
  return 0; // Non-zero return values are reserved for future use.
}
