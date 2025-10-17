/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

//===------------------------------------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_RUNTIME_LANGUAGE_EQUATABLE_SUPPORT_H
#define LANGUAGE_RUNTIME_LANGUAGE_EQUATABLE_SUPPORT_H

#include "language/Runtime/Metadata.h"
#include <stdint.h>

namespace language {
namespace equatable_support {

extern "C" const ProtocolDescriptor PROTOCOL_DESCR_SYM(SQ);
static constexpr auto &EquatableProtocolDescriptor = PROTOCOL_DESCR_SYM(SQ);

struct EquatableWitnessTable;

/// Calls `Equatable.==` through a `Equatable` (not Equatable!) witness
/// table.
LANGUAGE_CC(language) LANGUAGE_RUNTIME_STDLIB_INTERNAL
bool _language_stdlib_Equatable_isEqual_indirect(
    const void *lhsValue, const void *rhsValue, const Metadata *type,
    const EquatableWitnessTable *wt);

} // namespace equatable_support
} // namespace language

#endif

