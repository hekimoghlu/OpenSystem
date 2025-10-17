/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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

//===-- ForeignClassMetadataVisitor.h - foreign class metadata -*- C++ --*-===//
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
// A CRTP class useful for laying out foreign class metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_FOREIGNCLASSMETADATAVISITOR_H
#define LANGUAGE_IRGEN_FOREIGNCLASSMETADATAVISITOR_H

#include "NominalMetadataVisitor.h"

namespace language {
namespace irgen {

/// A CRTP layout class for foreign class metadata.
template <class Impl>
class ForeignClassMetadataVisitor
       : public NominalMetadataVisitor<Impl> {
  using super = NominalMetadataVisitor<Impl>;
protected:
  ClassDecl *Target;
  using super::asImpl;
public:
  ForeignClassMetadataVisitor(IRGenModule &IGM, ClassDecl *target)
    : super(IGM), Target(target) {}

  void layout() {
    asImpl().addLayoutStringPointer();
    super::layout();
    asImpl().addNominalTypeDescriptor();
    asImpl().addSuperclass();
    asImpl().addReservedWord();
  }

  CanType getTargetType() const {
    return Target->getDeclaredType()->getCanonicalType();
  }
};

/// An "implementation" of ForeignClassMetadataVisitor that just scans through
/// the metadata layout, maintaining the offset of the next field.
template <class Impl>
class ForeignClassMetadataScanner : public ForeignClassMetadataVisitor<Impl> {
  using super = ForeignClassMetadataVisitor<Impl>;

protected:
  Size NextOffset = Size(0);

  ForeignClassMetadataScanner(IRGenModule &IGM, ClassDecl *target)
    : super(IGM, target) {}

public:
  void addMetadataFlags() { addPointer(); }
  void addLayoutStringPointer() { addPointer(); }
  void addValueWitnessTable() { addPointer(); }
  void addNominalTypeDescriptor() { addPointer(); }
  void addSuperclass() { addPointer(); }
  void addReservedWord() { addPointer(); }

private:
  void addPointer() {
    NextOffset += super::IGM.getPointerSize();
  }
};

template <class Impl>
class ForeignReferenceTypeMetadataVisitor
    : public NominalMetadataVisitor<Impl> {
  using super = NominalMetadataVisitor<Impl>;
protected:
  ClassDecl *Target;
  using super::asImpl;
public:
  ForeignReferenceTypeMetadataVisitor(IRGenModule &IGM, ClassDecl *target)
      : super(IGM), Target(target) {}

  void layout() {
    asImpl().addLayoutStringPointer();
    super::layout();
    asImpl().addNominalTypeDescriptor();
    asImpl().addReservedWord();
  }

  CanType getTargetType() const {
    return Target->getDeclaredType()->getCanonicalType();
  }
};

} // end namespace irgen
} // end namespace language

#endif // LANGUAGE_IRGEN_FOREIGNCLASSMETADATAVISITOR_H
