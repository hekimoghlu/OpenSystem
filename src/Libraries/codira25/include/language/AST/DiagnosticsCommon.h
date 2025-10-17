/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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

//===--- DiagnosticsCommon.h - Shared Diagnostic Definitions ----*- C++ -*-===//
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
/// \file
/// This file defines common diagnostics for the whole compiler, as well
/// as some diagnostic infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_DIAGNOSTICSCOMMON_H
#define LANGUAGE_DIAGNOSTICSCOMMON_H

#include "language/AST/DiagnosticEngine.h"
#include "language/Basic/Toolchain.h"
#include "language/Config.h"

namespace language {
  class AccessorDecl;
  class ConstructorDecl;
  class MacroDecl;
  class SubscriptDecl;
  class SwitchStmt;
  class TypeAliasDecl;

  template<typename ...ArgTypes>
  struct Diag;

  namespace detail {
    // These templates are used to help extract the type arguments of the
    // DIAG/ERROR/WARNING/NOTE/REMARK/FIXIT macros.
    template<typename T>
    struct DiagWithArguments;
    
    template<typename ...ArgTypes>
    struct DiagWithArguments<void(ArgTypes...)> {
      typedef Diag<ArgTypes...> type;
    };

    template <typename T>
    struct StructuredFixItWithArguments;

    template <typename... ArgTypes>
    struct StructuredFixItWithArguments<void(ArgTypes...)> {
      typedef StructuredFixIt<ArgTypes...> type;
    };
  } // end namespace detail

  enum class StaticSpellingKind : uint8_t;
  enum class ForeignLanguage : uint8_t;

  namespace diag {

    enum class RequirementKind : uint8_t;

    using DeclAttribute = const DeclAttribute *;

  // Declare common diagnostics objects with their appropriate types.
#define DIAG(KIND, ID, Group, Options, Text, Signature)                    \
      extern detail::DiagWithArguments<void Signature>::type ID;
#define FIXIT(ID, Text, Signature)                                         \
      extern detail::StructuredFixItWithArguments<void Signature>::type ID;
#include "DiagnosticsCommon.def"
  } // end namespace diag
} // end namespace language

#endif
