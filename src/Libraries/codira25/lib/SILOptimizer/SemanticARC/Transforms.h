/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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

//===--- Transforms.h -----------------------------------------------------===//
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
///
/// \file
///
/// Top level transforms for SemanticARCOpts
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_SILOPTIMIZER_SEMANTICARCOPT_TRANSFORMS_H
#define LANGUAGE_SILOPTIMIZER_SEMANTICARCOPT_TRANSFORMS_H

#include "toolchain/Support/Compiler.h"

namespace language {
namespace semanticarc {

struct Context;

/// Given the current map of owned phi arguments to consumed incoming values in
/// ctx, attempt to convert these owned phi arguments to guaranteed phi
/// arguments if the phi arguments are the only thing that kept us from
/// converting these incoming values to be guaranteed.
///
/// \returns true if we converted atleast one phi from owned -> guaranteed and
/// eliminated ARC traffic as a result.
TOOLCHAIN_LIBRARY_VISIBILITY bool tryConvertOwnedPhisToGuaranteedPhis(Context &ctx);

} // namespace semanticarc
} // namespace language

#endif
