/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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

//===--- GenPoly.h - Codira IR generation for polymorphism -------*- C++ -*-===//
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
//  This file provides the private interface to the code for translating
//  between polymorphic and monomorphic values.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENPOLY_H
#define LANGUAGE_IRGEN_GENPOLY_H

namespace toolchain {
  class Type;
  template <class T> class ArrayRef;
}

namespace language {
  class CanType;

namespace irgen {
  class Explosion;
  class IRGenFunction;
  class IRGenModule;

  /// Do the given types differ by abstraction when laid out as memory?
  bool differsByAbstractionInMemory(IRGenModule &IGM,
                                    CanType origTy, CanType substTy);

  /// Do the given types differ by abstraction when passed in an explosion?
  bool differsByAbstractionInExplosion(IRGenModule &IGM,
                                       CanType origTy, CanType substTy);

  /// Given a substituted explosion, re-emit it as an unsubstituted one.
  ///
  /// For example, given an explosion which begins with the
  /// representation of an (Int, Float), consume that and produce the
  /// representation of an (Int, T).
  ///
  /// The substitutions must carry origTy to substTy.
  void reemitAsUnsubstituted(IRGenFunction &IGF,
                             SILType origTy, SILType substTy,
                             Explosion &src, Explosion &dest);

  /// True if a function's signature in LLVM carries polymorphic parameters.
  bool hasPolymorphicParameters(CanSILFunctionType ty);
} // end namespace irgen
} // end namespace language

#endif
