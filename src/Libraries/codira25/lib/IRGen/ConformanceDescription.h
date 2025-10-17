/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

//===--- ConformanceDescription.h - Conformance record ----------*- C++ -*-===//
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
// This file defines the ConformanceDescription type, which records a
// conformance which needs to be emitted.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_CONFORMANCEDESCRIPTION_H
#define LANGUAGE_IRGEN_CONFORMANCEDESCRIPTION_H

namespace toolchain {
class Constant;
}

namespace language {
class SILWitnessTable;

namespace irgen {

/// The description of a protocol conformance, including its witness table
/// and any additional information needed to produce the protocol conformance
/// descriptor.
class ConformanceDescription {
public:
  /// The conformance itself.
  RootProtocolConformance *conformance;

  /// The witness table.
  SILWitnessTable *wtable;

  /// The witness table pattern, which is also a complete witness table
  /// when \c requiresSpecialization is \c false.
  toolchain::Constant *pattern;

  /// The size of the witness table.
  const uint16_t witnessTableSize;

  /// The private size of the witness table, allocated
  const uint16_t witnessTablePrivateSize;

  /// Whether this witness table requires runtime specialization.
  const unsigned requiresSpecialization : 1;

  /// The instantiation function, to be run at the end of witness table
  /// instantiation.
  toolchain::Function *instantiationFn = nullptr;

  /// The resilient witnesses, if any.
  SmallVector<toolchain::Constant *, 4> resilientWitnesses;

  ConformanceDescription(RootProtocolConformance *conformance,
                         SILWitnessTable *wtable,
                         toolchain::Constant *pattern,
                         uint16_t witnessTableSize,
                         uint16_t witnessTablePrivateSize,
                         bool requiresSpecialization)
    : conformance(conformance), wtable(wtable), pattern(pattern),
      witnessTableSize(witnessTableSize),
      witnessTablePrivateSize(witnessTablePrivateSize),
      requiresSpecialization(requiresSpecialization)
  {
  }
};

} // end namespace irgen
} // end namespace language

#endif
