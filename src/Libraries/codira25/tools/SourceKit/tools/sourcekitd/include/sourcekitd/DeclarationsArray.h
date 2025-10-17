/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

//===--- DeclarationsArray.h - ----------------------------------*- C++ -*-===//
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
// This is an array used in the response to editor.open.interface requests.
// It contains all declarations, identified by their Kind, Offset, and Length,
// and optionally includes a USR, if the declaration has one.
//===----------------------------------------------------------------------===//
#ifndef TOOLCHAIN_SOURCEKITD_DECLARATIONS_ARRAY_H
#define TOOLCHAIN_SOURCEKITD_DECLARATIONS_ARRAY_H

#include "sourcekitd/Internal.h"

namespace sourcekitd {

VariantFunctions *getVariantFunctionsForDeclarationsArray();

/// Builds an array for declarations by kind, offset, length, and optionally USR
class DeclarationsArrayBuilder {
public:
  DeclarationsArrayBuilder();
  ~DeclarationsArrayBuilder();

  void add(SourceKit::UIdent Kind, unsigned Offset, unsigned Length,
           toolchain::StringRef USR);

  bool empty() const;

  std::unique_ptr<toolchain::MemoryBuffer> createBuffer();

private:
  struct Implementation;
  Implementation &Impl;
};

} // namespace sourcekitd

#endif
