/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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

//===--- Overrides.h - Compat overrides for Codira 5.0 runtime ------s------===//
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
//  This file provides compatibility override hooks for Codira 5.1 runtimes.
//
//===----------------------------------------------------------------------===//

#include "language/Runtime/Metadata.h"
#include "toolchain/ADT/StringRef.h"

namespace language {

using ConformsToCodiraProtocol_t =
  const ProtocolConformanceDescriptor *(const Metadata * const type,
                                        const ProtocolDescriptor *protocol,
                                        toolchain::StringRef moduleName);

const ProtocolConformanceDescriptor *
language51override_conformsToCodiraProtocol(const Metadata * const type,
                                        const ProtocolDescriptor *protocol,
                                        toolchain::StringRef moduleName,
                                        ConformsToCodiraProtocol_t *original);

} // end namespace language
