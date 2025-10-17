/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

//===--- ClangNode.h - How Codira tracks imported Clang entities -*- C++ -*-===//
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

#ifndef LANGUAGE_CLANGNODE_H
#define LANGUAGE_CLANGNODE_H

#include "language/Basic/Debug.h"
#include "toolchain/ADT/PointerUnion.h"

namespace language::Core {
  class Decl;
  class MacroInfo;
  class ModuleMacro;
  class Module;
  class SourceLocation;
  class SourceRange;
}

namespace language {
  
namespace detail {
  /// A wrapper to avoid having to import Clang headers. We can't just
  /// forward-declare their PointerLikeTypeTraits because we don't own
  /// the types.
  template <typename T>
  struct ClangNodeBox {
    const T * const value;

    ClangNodeBox() : value{nullptr} {}
    /*implicit*/ ClangNodeBox(const T *V) : value(V) {}

    explicit operator bool() const { return value; }
  };  
}

/// Represents a clang declaration, macro, or module. A macro definition
/// imported from a module is recorded as the ModuleMacro, and a macro
/// defined locally is represented by the MacroInfo.
class ClangNode {
  template <typename T>
  using Box = detail::ClangNodeBox<T>;

  toolchain::PointerUnion<Box<language::Core::Decl>, Box<language::Core::MacroInfo>,
                     Box<language::Core::ModuleMacro>, Box<language::Core::Module>> Ptr;

public:
  ClangNode() = default;
  ClangNode(const language::Core::Decl *D) : Ptr(D) {}
  ClangNode(const language::Core::MacroInfo *MI) : Ptr(MI) {}
  ClangNode(const language::Core::ModuleMacro *MM) : Ptr(MM) {}
  ClangNode(const language::Core::Module *Mod) : Ptr(Mod) {}

  bool isNull() const { return Ptr.isNull(); }
  explicit operator bool() const { return !isNull(); }

  const language::Core::Decl *getAsDecl() const {
    return Ptr.dyn_cast<Box<language::Core::Decl>>().value;
  }
  const language::Core::MacroInfo *getAsMacroInfo() const {
    return Ptr.dyn_cast<Box<language::Core::MacroInfo>>().value;
  }
  const language::Core::ModuleMacro *getAsModuleMacro() const {
    return Ptr.dyn_cast<Box<language::Core::ModuleMacro>>().value;
  }
  const language::Core::Module *getAsModule() const {
    return Ptr.dyn_cast<Box<language::Core::Module>>().value;
  }

  const language::Core::Decl *castAsDecl() const {
    return Ptr.get<Box<language::Core::Decl>>().value;
  }
  const language::Core::MacroInfo *castAsMacroInfo() const {
    return Ptr.get<Box<language::Core::MacroInfo>>().value;
  }
  const language::Core::ModuleMacro *castAsModuleMacro() const {
    return Ptr.get<Box<language::Core::ModuleMacro>>().value;
  }
  const language::Core::Module *castAsModule() const {
    return Ptr.get<Box<language::Core::Module>>().value;
  }

  // Get the MacroInfo for a local definition, one imported from a
  // ModuleMacro, or null if it's neither.
  const language::Core::MacroInfo *getAsMacro() const;

  /// Returns the module either the one wrapped directly, the one from a
  /// language::Core::ImportDecl or null if it's neither.
  const language::Core::Module *getClangModule() const;

  /// Returns the owning clang module of this node, if it exists.
  const language::Core::Module *getOwningClangModule() const;

  language::Core::SourceLocation getLocation() const;
  language::Core::SourceRange getSourceRange() const;

  LANGUAGE_DEBUG_DUMP;

  void *getOpaqueValue() const { return Ptr.getOpaqueValue(); }
  static inline ClangNode getFromOpaqueValue(void *VP) {
    ClangNode N;
    N.Ptr = decltype(Ptr)::getFromOpaqueValue(VP);
    return N;
  }
};

} // end namespace language

namespace toolchain {
template <typename T>
struct PointerLikeTypeTraits<language::detail::ClangNodeBox<T>> {
  using Box = ::language::detail::ClangNodeBox<T>;

  static inline void *getAsVoidPointer(Box P) {
    return const_cast<void *>(static_cast<const void *>(P.value));
  }
  static inline Box getFromVoidPointer(const void *P) {
    return Box{static_cast<const T *>(P)};
  }

  /// Note: We are relying on Clang nodes to be at least 4-byte aligned.
  enum { NumLowBitsAvailable = 2 };
};
  
} // end namespace toolchain

#endif
