/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

//===--- DocStructureArray.h - ----------------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_SOURCEKITD_DOC_STRUCTURE_ARRAY_H
#define TOOLCHAIN_SOURCEKITD_DOC_STRUCTURE_ARRAY_H

#include "sourcekitd/Internal.h"
#include "toolchain/ADT/SmallString.h"

namespace sourcekitd {

VariantFunctions *getVariantFunctionsForDocStructureArray();
VariantFunctions *getVariantFunctionsForDocStructureElementArray();
VariantFunctions *getVariantFunctionsForInheritedTypesArray();
VariantFunctions *getVariantFunctionsForAttributesArray();

class DocStructureArrayBuilder {
public:
  DocStructureArrayBuilder();
  ~DocStructureArrayBuilder();

  void beginSubStructure(unsigned Offset, unsigned Length,
                         SourceKit::UIdent Kind, SourceKit::UIdent AccessLevel,
                         SourceKit::UIdent SetterAccessLevel,
                         unsigned NameOffset, unsigned NameLength,
                         unsigned BodyOffset, unsigned BodyLength,
                         unsigned DocOffset, unsigned DocLength,
                         toolchain::StringRef DisplayName, toolchain::StringRef TypeName,
                         toolchain::StringRef RuntimeName,
                         toolchain::StringRef SelectorName,
                         toolchain::ArrayRef<toolchain::StringRef> InheritedTypes,
                         toolchain::ArrayRef<std::tuple<SourceKit::UIdent, unsigned, unsigned>> Attrs);

  void addElement(SourceKit::UIdent Kind, unsigned Offset, unsigned Length);

  void endSubStructure();

  std::unique_ptr<toolchain::MemoryBuffer> createBuffer();

private:
  struct Implementation;
  Implementation &impl;
};

} // end namespace sourcekitd

#endif // TOOLCHAIN_SOURCEKITD_DOC_STRUCTURE_ARRAY_H
