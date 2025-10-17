/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

//===--- TargetInfo.h - Target Info Output ---------------------*- C++ -*-===//
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
// This file provides a high-level API for emitting target info
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_TARGETINFO_H
#define LANGUAGE_TARGETINFO_H

#include "language/Basic/Toolchain.h"

#include <optional>

namespace toolchain {
class Triple;
class VersionTuple;
}

namespace language {
class CompilerInvocation;

namespace targetinfo {
void printTargetInfo(const CompilerInvocation &invocation,
                     toolchain::raw_ostream &out);

void printTripleInfo(const CompilerInvocation &invocation,
                     const toolchain::Triple &triple,
                     std::optional<toolchain::VersionTuple> runtimeVersion,
                     toolchain::raw_ostream &out);
} // namespace targetinfo
} // namespace language

#endif
