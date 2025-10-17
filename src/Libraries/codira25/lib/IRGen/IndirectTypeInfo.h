/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

//===--- IndirectTypeInfo.h - Convenience for indirected types --*- C++ -*-===//
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
// This file defines IndirectTypeInfo, which is a convenient abstract
// implementation of TypeInfo for working with types that are always
// passed or returned indirectly.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_INDIRECTTYPEINFO_H
#define LANGUAGE_IRGEN_INDIRECTTYPEINFO_H

#include "Explosion.h"
#include "TypeInfo.h"
#include "IRGenFunction.h"

namespace language {
namespace irgen {

/// IndirectTypeInfo - An abstract class designed for use when
/// implementing a type which is always passed indirectly.
///
/// Subclasses must implement the following operations:
///   allocateStack
///   assignWithCopy
///   initializeWithCopy
///   destroy
template <class Derived, class Base>
class IndirectTypeInfo : public Base {
protected:
  template <class... T> IndirectTypeInfo(T &&...args)
    : Base(::std::forward<T>(args)...) {}

  const Derived &asDerived() const {
    return static_cast<const Derived &>(*this);
  }

public:
  void getSchema(ExplosionSchema &schema) const override {
    schema.add(ExplosionSchema::Element::forAggregate(this->getStorageType(),
                                              this->getBestKnownAlignment()));
  }

  void initializeFromParams(IRGenFunction &IGF, Explosion &params, Address dest,
                            SILType T, bool isOutlined) const override {
    Address src = this->getAddressForPointer(params.claimNext());
    asDerived().Derived::initializeWithTake(IGF, dest, src, T, isOutlined,
                                            /*zeroizeIfSensitive=*/ true);
  }

  void assignWithTake(IRGenFunction &IGF, Address dest, Address src, SILType T,
                      bool isOutlined) const override {
    asDerived().Derived::destroy(IGF, dest, T, isOutlined);
    asDerived().Derived::initializeWithTake(IGF, dest, src, T, isOutlined,
                                            /*zeroizeIfSensitive=*/ true);
  }
};

}
}

#endif
