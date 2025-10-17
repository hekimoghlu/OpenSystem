/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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

//===--- TokenAnnotationsArray.h - ------------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_SOURCEKITD_TOKEN_ANNOTATIONS_ARRAY_H
#define TOOLCHAIN_SOURCEKITD_TOKEN_ANNOTATIONS_ARRAY_H

#include "sourcekitd/Internal.h"

namespace sourcekitd {

VariantFunctions *getVariantFunctionsForTokenAnnotationsArray();

class TokenAnnotationsArrayBuilder {
public:
  TokenAnnotationsArrayBuilder();
  ~TokenAnnotationsArrayBuilder();

  void add(SourceKit::UIdent Kind,
           unsigned Offset,
           unsigned Length,
           bool IsSystem);

  bool empty() const;

  std::unique_ptr<toolchain::MemoryBuffer> createBuffer();

private:
  struct Implementation;
  Implementation &Impl;
};

}

#endif
