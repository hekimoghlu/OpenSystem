/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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

//===--- ClassTypeInfo.h - The layout info for class types. -----*- C++ -*-===//
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
//  This file contains layout information for class types.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_CLASSTYPEINFO_H
#define LANGUAGE_IRGEN_CLASSTYPEINFO_H

#include "ClassLayout.h"
#include "HeapTypeInfo.h"

#include "language/ClangImporter/ClangImporterRequests.h"

namespace language {
namespace irgen {

/// Layout information for class types.
class ClassTypeInfo : public HeapTypeInfo<ClassTypeInfo> {
  ClassDecl *TheClass;
  mutable toolchain::StructType *classLayoutType;

  // The resilient layout of the class, without making any assumptions
  // that violate resilience boundaries. This is used to allocate
  // and deallocate instances of the class, and to access fields.
  mutable std::optional<ClassLayout> ResilientLayout;

  // A completely fragile layout, used for metadata emission.
  mutable std::optional<ClassLayout> FragileLayout;

  /// Can we use language reference-counting, or do we have to use
  /// objc_retain/release?
  const ReferenceCounting Refcount;

  ClassLayout generateLayout(IRGenModule &IGM, SILType classType,
                             bool forBackwardDeployment) const;

public:
  ClassTypeInfo(toolchain::PointerType *irType, Size size, SpareBitVector spareBits,
                Alignment align, ClassDecl *theClass,
                ReferenceCounting refcount, toolchain::StructType *classLayoutType)
      : HeapTypeInfo(refcount, irType, size, std::move(spareBits), align),
        TheClass(theClass), classLayoutType(classLayoutType),
        Refcount(refcount) {}

  ReferenceCounting getReferenceCounting() const { return Refcount; }

  ClassDecl *getClass() const { return TheClass; }

  toolchain::Type *getClassLayoutType() const { return classLayoutType; }

  const ClassLayout &getClassLayout(IRGenModule &IGM, SILType type,
                                    bool forBackwardDeployment) const;

  StructLayout *createLayoutWithTailElems(IRGenModule &IGM, SILType classType,
                                          ArrayRef<SILType> tailTypes) const;

  void emitScalarRelease(IRGenFunction &IGF, toolchain::Value *value,
                         Atomicity atomicity) const override {
    if (getReferenceCounting() == ReferenceCounting::Custom) {
      auto releaseFn =
          evaluateOrDefault(
              getClass()->getASTContext().evaluator,
              CustomRefCountingOperation(
                  {getClass(), CustomRefCountingOperationKind::release}),
              {})
              .operation;
      IGF.emitForeignReferenceTypeLifetimeOperation(releaseFn, value);
      return;
    }

    HeapTypeInfo::emitScalarRelease(IGF, value, atomicity);
  }

  void emitScalarRetain(IRGenFunction &IGF, toolchain::Value *value,
                        Atomicity atomicity) const override {
    if (getReferenceCounting() == ReferenceCounting::Custom) {
      auto retainFn =
          evaluateOrDefault(
              getClass()->getASTContext().evaluator,
              CustomRefCountingOperation(
                  {getClass(), CustomRefCountingOperationKind::retain}),
              {})
              .operation;
      IGF.emitForeignReferenceTypeLifetimeOperation(retainFn, value);
      return;
    }

    HeapTypeInfo::emitScalarRetain(IGF, value, atomicity);
  }

  void strongCustomRetain(IRGenFunction &IGF, Explosion &e,
                          bool needsNullCheck) const {
    assert(getReferenceCounting() == ReferenceCounting::Custom &&
           "only supported for custom ref-counting");

    toolchain::Value *value = e.claimNext();
    auto retainFn =
        evaluateOrDefault(
            getClass()->getASTContext().evaluator,
            CustomRefCountingOperation(
                {getClass(), CustomRefCountingOperationKind::retain}),
            {})
            .operation;
    IGF.emitForeignReferenceTypeLifetimeOperation(retainFn, value,
                                                  needsNullCheck);
  }

  // Implement the primary retain/release operations of ReferenceTypeInfo
  // using basic reference counting.
  void strongRetain(IRGenFunction &IGF, Explosion &e,
                    Atomicity atomicity) const override {
    if (getReferenceCounting() == ReferenceCounting::Custom) {
      strongCustomRetain(IGF, e, /*needsNullCheck*/ false);
      return;
    }

    HeapTypeInfo::strongRetain(IGF, e, atomicity);
  }

  void strongCustomRelease(IRGenFunction &IGF, Explosion &e,
                           bool needsNullCheck) const {
    assert(getReferenceCounting() == ReferenceCounting::Custom &&
           "only supported for custom ref-counting");

    toolchain::Value *value = e.claimNext();
    auto releaseFn =
        evaluateOrDefault(
            getClass()->getASTContext().evaluator,
            CustomRefCountingOperation(
                {getClass(), CustomRefCountingOperationKind::release}),
            {})
            .operation;
    IGF.emitForeignReferenceTypeLifetimeOperation(releaseFn, value,
                                                  needsNullCheck);
  }

  void strongRelease(IRGenFunction &IGF, Explosion &e,
                     Atomicity atomicity) const override {
    if (getReferenceCounting() == ReferenceCounting::Custom) {
      strongCustomRelease(IGF, e, /*needsNullCheck*/ false);
      return;
    }

    HeapTypeInfo::strongRelease(IGF, e, atomicity);
  }
};

} // namespace irgen
} // namespace language

#endif
