/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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

//=== SelectorExtras.h - Helpers for checkers using selectors -----*- C++ -*-=//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_ANALYSIS_SELECTOREXTRAS_H
#define LANGUAGE_CORE_ANALYSIS_SELECTOREXTRAS_H

#include "language/Core/AST/ASTContext.h"

namespace language::Core {

template <typename... IdentifierInfos>
static inline Selector getKeywordSelector(ASTContext &Ctx,
                                          const IdentifierInfos *...IIs) {
  static_assert(sizeof...(IdentifierInfos) > 0,
                "keyword selectors must have at least one argument");
  SmallVector<const IdentifierInfo *, 10> II({&Ctx.Idents.get(IIs)...});

  return Ctx.Selectors.getSelector(II.size(), &II[0]);
}

template <typename... IdentifierInfos>
static inline void lazyInitKeywordSelector(Selector &Sel, ASTContext &Ctx,
                                           IdentifierInfos *... IIs) {
  if (!Sel.isNull())
    return;
  Sel = getKeywordSelector(Ctx, IIs...);
}

} // end namespace language::Core

#endif
