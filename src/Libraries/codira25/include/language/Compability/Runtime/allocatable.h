/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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

//===-- language/Compability/Runtime/allocatable.h ---------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// Author: Tunjay Akbarli
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

// Defines APIs for Fortran runtime library support of code generated
// to manipulate and query allocatable variables, dummy arguments, & components.
#ifndef LANGUAGE_COMPABILITY_RUNTIME_ALLOCATABLE_H_
#define LANGUAGE_COMPABILITY_RUNTIME_ALLOCATABLE_H_

#include "language/Compability/Runtime/descriptor-consts.h"
#include "language/Compability/Runtime/entry-names.h"

namespace language::Compability::runtime {

extern "C" {

// Initializes the descriptor for an allocatable of intrinsic or derived type.
// The incoming descriptor is treated as (and can be) uninitialized garbage.
// Must be called for each allocatable variable as its scope comes into being.
// The storage for the allocatable's descriptor must have already been
// allocated to a size sufficient for the rank, corank, and type.
// A descriptor must be initialized before being used for any purpose,
// but needs reinitialization in a deallocated state only when there is
// a change of type, rank, or corank.
void RTDECL(AllocatableInitIntrinsic)(
    Descriptor &, TypeCategory, int kind, int rank = 0, int corank = 0);
void RTDECL(AllocatableInitCharacter)(Descriptor &, SubscriptValue length = 0,
    int kind = 1, int rank = 0, int corank = 0);
void RTDECL(AllocatableInitDerived)(
    Descriptor &, const typeInfo::DerivedType &, int rank = 0, int corank = 0);

// Initializes the descriptor for an allocatable of intrinsic or derived type.
// These functions are meant to be used in the allocate statement lowering. If
// the descriptor is allocated, the initialization is skiped so the error
// handling can be done by AllocatableAllocate.
void RTDECL(AllocatableInitIntrinsicForAllocate)(
    Descriptor &, TypeCategory, int kind, int rank = 0, int corank = 0);
void RTDECL(AllocatableInitCharacterForAllocate)(Descriptor &,
    SubscriptValue length = 0, int kind = 1, int rank = 0, int corank = 0);
void RTDECL(AllocatableInitDerivedForAllocate)(
    Descriptor &, const typeInfo::DerivedType &, int rank = 0, int corank = 0);

// Checks that an allocatable is not already allocated in statements
// with STAT=.  Use this on a value descriptor before setting bounds or
// type parameters.  Not necessary on a freshly initialized descriptor.
// (If there's no STAT=, the error will be caught later anyway, but
// this API allows the error to be caught before descriptor is modified.)
// Return 0 on success (deallocated state), else the STAT= value.
int RTDECL(AllocatableCheckAllocated)(Descriptor &,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// For MOLD= allocation; sets bounds, cobounds, and length type
// parameters from another descriptor. The destination descriptor must
// be initialized and deallocated.
void RTDECL(AllocatableApplyMold)(
    Descriptor &, const Descriptor &mold, int rank = 0);

// Explicitly sets the bounds and length type parameters of an initialized
// deallocated allocatable.
void RTDECL(AllocatableSetBounds)(
    Descriptor &, int zeroBasedDim, SubscriptValue lower, SubscriptValue upper);

// The upper cobound is ignored for the last codimension.
void RTDECL(AllocatableSetCoBounds)(Descriptor &, int zeroBasedCoDim,
    SubscriptValue lower, SubscriptValue upper = 0);

// Length type parameters are indexed in declaration order; i.e., 0 is the
// first length type parameter in the deepest base type.  (Not for use
// with CHARACTER; see above.)
void RTDECL(AllocatableSetDerivedLength)(
    Descriptor &, int which, SubscriptValue);

// When an explicit type-spec appears in an ALLOCATE statement for an
// allocatable with an explicit (non-deferred) length type paramater for
// a derived type or CHARACTER value, the explicit value has to match
// the length type parameter's value.  This API checks that requirement.
// Returns 0 for success, or the STAT= value on failure with hasStat==true.
int RTDECL(AllocatableCheckLengthParameter)(Descriptor &,
    int which /* 0 for CHARACTER length */, SubscriptValue other,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Allocates an allocatable.  The allocatable descriptor must have been
// initialized and its bounds and length type parameters set and must be
// in a deallocated state.
// On failure, if hasStat is true, returns a nonzero error code for
// STAT= and (if present) fills in errMsg; if hasStat is false, the
// image is terminated.  On success, leaves errMsg alone and returns zero.
// Successfully allocated memory is initialized if the allocatable has a
// derived type, and is always initialized by AllocatableAllocateSource().
// Performs all necessary coarray synchronization and validation actions.
int RTDECL(AllocatableAllocate)(Descriptor &,
    std::int64_t *asyncObject = nullptr, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);
int RTDECL(AllocatableAllocateSource)(Descriptor &, const Descriptor &source,
    bool hasStat = false, const Descriptor *errMsg = nullptr,
    const char *sourceFile = nullptr, int sourceLine = 0);

// Implements the intrinsic subroutine MOVE_ALLOC (16.9.137 in F'2018,
// but note the order of first two arguments is reversed for consistency
// with the other APIs for allocatables.)  The destination descriptor
// must be initialized.
std::int32_t RTDECL(MoveAlloc)(Descriptor &to, Descriptor &from,
    const typeInfo::DerivedType *, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Deallocates an allocatable.  Finalizes elements &/or components as needed.
// The allocatable is left in an initialized state suitable for reallocation
// with the same bounds, cobounds, and length type parameters.
int RTDECL(AllocatableDeallocate)(Descriptor &, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Same as AllocatableDeallocate but also set the dynamic type as the declared
// type as mentioned in 7.3.2.3 note 7.
int RTDECL(AllocatableDeallocatePolymorphic)(Descriptor &,
    const typeInfo::DerivedType *, bool hasStat = false,
    const Descriptor *errMsg = nullptr, const char *sourceFile = nullptr,
    int sourceLine = 0);

// Variant of above that does not finalize; for intermediate results
void RTDECL(AllocatableDeallocateNoFinal)(
    Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
} // extern "C"
} // namespace language::Compability::runtime
#endif // FORTRAN_RUNTIME_ALLOCATABLE_H_
