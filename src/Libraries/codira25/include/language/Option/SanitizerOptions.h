/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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

//===--- SanitizerOptions.h - Helpers related to sanitizers -----*- C++ -*-===//
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

#ifndef LANGUAGE_OPTIONS_SANITIZER_OPTIONS_H
#define LANGUAGE_OPTIONS_SANITIZER_OPTIONS_H

#include "language/Basic/Sanitizers.h"
#include "language/Basic/OptionSet.h"
#include "toolchain/TargetParser/Triple.h"
#include "toolchain/Option/Arg.h"
#include "toolchain/Option/ArgList.h"
// FIXME: This include is just for toolchain::SanitizerCoverageOptions. We should
// split the header upstream so we don't include so much.
#include "toolchain/Transforms/Instrumentation.h"

namespace language {
class DiagnosticEngine;

/// Parses a -sanitize= argument's values.
///
/// \param Diag If non null, the argument is used to diagnose invalid values.
/// \param sanitizerRuntimeLibExists Function which checks for existence of a
//         sanitizer dylib with a given name.
/// \return Returns a SanitizerKind.
OptionSet<SanitizerKind> parseSanitizerArgValues(
    const toolchain::opt::ArgList &Args, const toolchain::opt::Arg *A,
    const toolchain::Triple &Triple, DiagnosticEngine &Diag,
    toolchain::function_ref<bool(toolchain::StringRef, bool)> sanitizerRuntimeLibExists);

OptionSet<SanitizerKind> parseSanitizerRecoverArgValues(
    const toolchain::opt::Arg *A, const OptionSet<SanitizerKind> &enabledSanitizers,
    DiagnosticEngine &Diags, bool emitWarnings);

/// Parses a -sanitize-coverage= argument's value.
toolchain::SanitizerCoverageOptions parseSanitizerCoverageArgValue(
        const toolchain::opt::Arg *A,
        const toolchain::Triple &Triple,
        DiagnosticEngine &Diag,
        OptionSet<SanitizerKind> sanitizers);

/// Parse -sanitize-address-use-odr-indicator's value.
bool parseSanitizerAddressUseODRIndicator(
    const toolchain::opt::Arg *A, const OptionSet<SanitizerKind> &enabledSanitizers,
    DiagnosticEngine &Diags);

/// Parse -sanitize-stable-abi's value
bool parseSanitizerUseStableABI(
    const toolchain::opt::Arg *A, const OptionSet<SanitizerKind> &enabledSanitizers,
    DiagnosticEngine &Diags);

/// Returns the active sanitizers as a comma-separated list.
std::string getSanitizerList(const OptionSet<SanitizerKind> &Set);
}
#endif // LANGUAGE_OPTIONS_SANITIZER_OPTIONS_H
