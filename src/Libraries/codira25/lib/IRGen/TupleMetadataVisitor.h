/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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

//===--- TupleMetadataVisitor.h - CRTP for tuple metadata ------*- C++ -*-===//
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
// A CRTP class useful for laying out tuple metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_TUPLEMETADATALAYOUT_H
#define LANGUAGE_IRGEN_TUPLEMETADATALAYOUT_H

//#include "NominalMetadataVisitor.h"

namespace language {
namespace irgen {

/// A CRTP class for laying out tuple metadata.
///
/// This produces an object corresponding to a TupleTypeMetadata type.
/// It does not itself doing anything special for metadata templates.
template <class Impl> struct TupleMetadataVisitor
       : public MetadataVisitor<Impl> {
  using super = MetadataVisitor<Impl>;

protected:
  using super::asImpl;

  TupleType *const Target;

  TupleMetadataVisitor(IRGenModule &IGM, TupleType *const target)
    : super(IGM), Target(target) {}

public:
  void layout() {
    super::layout();

    asImpl().addNumElementsInfo();
    asImpl().addLabelsInfo();

    for (unsigned i = 0, n = Target->getNumElements(); i < n; ++i) {
      asImpl().addElement(i, Target->getElement(i));
    }
  }
};

/// An "implementation" of TupleMetadataVisitor that just scans through
/// the metadata layout, maintaining the offset of all tuple elements.
template <class Impl> class TupleMetadataScanner
       : public TupleMetadataVisitor<Impl> {
  using super = TupleMetadataVisitor<Impl>;

protected:
  Size NextOffset = Size(0);

  TupleMetadataScanner(IRGenModule &IGM, TupleType *const target)
    : super(IGM, target) {}

public:
  void addValueWitnessTable() { addPointer(); }
  void addMetadataFlags() { addPointer(); }
  void addNumElementsInfo() { addPointer(); }
  void addLabelsInfo() { addPointer(); }
  void addElement(unsigned idx,
                  const TupleTypeElt &e) { addPointer(); addPointer(); }

private:
  void addPointer() { NextOffset += super::IGM.getPointerSize(); }
};

} // end namespace irgen
} // end namespace language

#endif
