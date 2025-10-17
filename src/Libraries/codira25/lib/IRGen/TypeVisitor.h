/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

//===--- TypeVisitor.h - IR-gen TypeVisitor specialization ------*- C++ -*-===//
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
// This file defines various type visitors that are useful in
// IR-generation.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_TYPEVISITOR_H
#define LANGUAGE_IRGEN_TYPEVISITOR_H

#include "language/AST/CanTypeVisitor.h"

namespace language {
namespace irgen {

/// ReferenceTypeVisitor - This is a specialization of CanTypeVisitor
/// which automatically ignores non-reference types.
template <typename ImplClass, typename RetTy = void, typename... Args>
class ReferenceTypeVisitor : public CanTypeVisitor<ImplClass, RetTy, Args...> {
#define TYPE(Id) \
  RetTy visit##Id##Type(Can##Id##Type T, Args... args) { \
    toolchain_unreachable(#Id "Type is not a reference type"); \
  }
  TYPE(BoundGenericEnum)
  TYPE(BoundGenericStruct)
  TYPE(BuiltinFloat)
  TYPE(BuiltinInteger)
  TYPE(BuiltinRawPointer)
  TYPE(BuiltinVector)
  TYPE(LValue)
  TYPE(Metatype)
  TYPE(Module)
  TYPE(Enum)
  TYPE(ReferenceStorage)
  TYPE(Struct)
  TYPE(Tuple)
#undef TYPE

  // BuiltinNativeObject
  // BuiltinBridgeObject
  // Class
  // BoundGenericClass
  // Protocol
  // ProtocolComposition
  // Archetype
  // Function
};
  
} // end namespace irgen
} // end namespace language
  
#endif
