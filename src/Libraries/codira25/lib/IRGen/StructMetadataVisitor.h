/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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

//===--- StructMetadataVisitor.h - CRTP for struct metadata ------*- C++ -*-===//
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
// A CRTP class useful for laying out struct metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_STRUCTMETADATALAYOUT_H
#define LANGUAGE_IRGEN_STRUCTMETADATALAYOUT_H

#include "Field.h"
#include "NominalMetadataVisitor.h"
#include "language/AST/IRGenOptions.h"

namespace language {
namespace irgen {

/// A CRTP class for laying out struct metadata.
///
/// This produces an object corresponding to the StructMetadata type.
/// It does not itself doing anything special for metadata templates.
template <class Impl> class StructMetadataVisitor
       : public NominalMetadataVisitor<Impl> {
  using super = NominalMetadataVisitor<Impl>;

protected:
  using super::IGM;
  using super::asImpl;

  /// The struct.
  StructDecl *const Target;

  StructMetadataVisitor(IRGenModule &IGM, StructDecl *target)
    : super(IGM), Target(target) {}

public:
  void layout() {
    static_assert(MetadataAdjustmentIndex::ValueType == 2,
                  "Adjustment index must be synchronized with this layout");

    asImpl().addLayoutStringPointer();

    // Metadata header.
    super::layout();

    // StructMetadata header.
    asImpl().addNominalTypeDescriptor();

    // Everything after this is type-specific.
    asImpl().noteStartOfTypeSpecificMembers();

    // Generic arguments.
    // This must always be the first piece of trailing data.
    asImpl().addGenericFields(Target);

    // Struct field offsets.
    asImpl().noteStartOfFieldOffsets();
    for (VarDecl *prop : Target->getStoredProperties()) {
      if (isExportableField(prop))
        asImpl().addFieldOffset(prop);
    }

    asImpl().noteEndOfFieldOffsets();

    if (asImpl().hasTrailingFlags())
      asImpl().addTrailingFlags();
  }
  
  // Note the start of the field offset vector.
  void noteStartOfFieldOffsets() {}

  // Note the end of the field offset vector.
  void noteEndOfFieldOffsets() {}

  bool hasTrailingFlags() {
    return Target->isGenericContext() &&
           IGM.shouldPrespecializeGenericMetadata();
  }
};

/// An "implementation" of StructMetadataVisitor that just scans through
/// the metadata layout, maintaining the offset of the next field.
template <class Impl>
class StructMetadataScanner : public StructMetadataVisitor<Impl> {
  using super = StructMetadataVisitor<Impl>;

protected:
  Size NextOffset = Size(0);

  StructMetadataScanner(IRGenModule &IGM, StructDecl *target)
    : super(IGM, target) {}

public:
  void addMetadataFlags() { addPointer(); }
  void addLayoutStringPointer() { addPointer(); }
  void addValueWitnessTable() { addPointer(); }
  void addNominalTypeDescriptor() { addPointer(); }
  void addFieldOffset(VarDecl *) { addInt32(); }
  void addGenericRequirement(GenericRequirement requirement) { addPointer(); }
  void noteStartOfTypeSpecificMembers() {}

  void noteEndOfFieldOffsets() {
    NextOffset = NextOffset.roundUpToAlignment(super::IGM.getPointerAlignment());
  }

  void addTrailingFlags() { addInt64(); }

private:
  void addPointer() {
    NextOffset += super::IGM.getPointerSize();
  }
  void addInt32() { NextOffset += Size(4); }
  void addInt64() { NextOffset += Size(8); }
};

} // end namespace irgen
} // end namespace language

#endif
