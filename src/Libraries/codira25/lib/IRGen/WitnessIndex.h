/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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

//===--- WitnessIndex.h - Index into a witness table ------------*- C++ -*-===//
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
// This file defines the WitnessIndex type, used for drilling into a
// protocol witness table or value witness table.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_WITNESSINDEX_H
#define LANGUAGE_IRGEN_WITNESSINDEX_H

#include "language/ABI/MetadataValues.h"
#include "language/IRGen/ValueWitness.h"

namespace language {
namespace irgen {

/// A class which encapsulates an index into a witness table.
class WitnessIndex {
  // Negative values are indexing into the private area of a protocol witness
  // table.
  int Value : 31;
  unsigned IsPrefix : 1;
public:
  WitnessIndex() = default;
  explicit WitnessIndex(int index, bool isPrefix)
      : Value(index), IsPrefix(isPrefix) {}

  int getValue() const { return Value; }

  bool isPrefix() const { return IsPrefix; }

  /// Adjust the index to refer into a protocol witness table (rather than
  /// a value witness table).
  WitnessIndex forProtocolWitnessTable() const {
    int NewValue = Value < 0
                     ? Value
                     : Value + WitnessTableFirstRequirementOffset;
    return WitnessIndex(NewValue, IsPrefix);
  }
};

} // end namespace irgen
} // end namespace language

#endif
