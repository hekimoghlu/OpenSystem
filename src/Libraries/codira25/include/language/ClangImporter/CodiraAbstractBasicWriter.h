/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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

//===- CodiraAbstractBasicWriter.h - Clang serialization adapter -*- C++ -*-===//
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
// This file provides an intermediate CRTP class which implements most of
// Clang's AbstractBasicWriter interface, allowing largely the same logic
// to be used for both the importer's "can this be serialized" checks and
// the serializer's actual serialization logic.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CLANGIMPORTER_LANGUAGEABSTRACTBASICWRITER_H
#define LANGUAGE_CLANGIMPORTER_LANGUAGEABSTRACTBASICWRITER_H

#include "language/Core/AST/AbstractTypeWriter.h"
#include "language/Core/AST/Type.h"

namespace language {

/// An implementation of Clang's AbstractBasicWriter interface for a Codira
/// datastream-based reader.  This is paired with the AbstractBasicReader
/// implementation in CodiraAbstractBasicReader.h.  Note that the general
/// expectation is that the types and declarations involved will have passed
/// a serializability check when this is used for actual serialization.
/// The code in this class is also used when implementing that
/// serializability check and so must be a little more cautious.
///
/// The subclass must implement:
///   void writeUInt64(uint64_t value);
///   void writeIdentifier(const language::Core::IdentifierInfo *ident);
///   void writeStmtRef(const language::Core::Stmt *stmt);
///   void writeDeclRef(const language::Core::Decl *decl);
template <class Impl>
class DataStreamBasicWriter
       : public language::Core::serialization::DataStreamBasicWriter<Impl> {
  using super = language::Core::serialization::DataStreamBasicWriter<Impl>;
public:
  using super::asImpl;

  DataStreamBasicWriter(language::Core::ASTContext &ctx) : super(ctx) {}

  /// Perform all the calls necessary to write out the given type.
  void writeTypeRef(const language::Core::Type *type) {
    asImpl().writeUInt64(uint64_t(type->getTypeClass()));
    language::Core::serialization::AbstractTypeWriter<Impl>(asImpl()).write(type);
  }

  void writeBool(bool value) {
    asImpl().writeUInt64(uint64_t(value));
  }

  void writeUInt32(uint32_t value) {
    asImpl().writeUInt64(uint64_t(value));
  }

  void writeSelector(language::Core::Selector selector) {
    if (selector.isNull()) {
      asImpl().writeUInt64(0);
      return;
    }

    asImpl().writeUInt64(selector.getNumArgs() + 1);
    for (unsigned i = 0, e = std::max(selector.getNumArgs(), 1U); i != e; ++i)
      asImpl().writeIdentifier(selector.getIdentifierInfoForSlot(i));
  }

  void writeSourceLocation(language::Core::SourceLocation loc) {
    // DataStreamBasicReader will always read null; the serializability
    // check overrides this to complain about non-null source locations.
  }

  void writeQualType(language::Core::QualType type) {
    assert(!type.isNull());

    auto split = type.split();
    auto qualifiers = split.Quals;

    // Unwrap BTFTagAttributeType and merge any of its qualifiers.
    while (auto btfType = dyn_cast<language::Core::BTFTagAttributedType>(split.Ty)) {
      split = btfType->getWrappedType().split();
      qualifiers.addQualifiers(split.Quals);
    }

    asImpl().writeQualifiers(qualifiers);
    // Just recursively visit the given type.
    asImpl().writeTypeRef(split.Ty);
  }

  void writeBTFTypeTagAttr(const language::Core::BTFTypeTagAttr *attr) {
    // BTFTagAttributeType is explicitly unwrapped above, so we should never
    // hit any of its attributes.
    toolchain::report_fatal_error("Should never hit BTFTypeTagAttr serialization");
  }
};

}

#endif
