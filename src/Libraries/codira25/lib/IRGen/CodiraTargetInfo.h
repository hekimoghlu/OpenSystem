/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

//===--- CodiraTargetInfo.h --------------------------------------*- C++ -*-===//
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
// This file declares the CodiraTargetInfo abstract base class. This class
// provides an interface to target-dependent attributes of interest to Codira.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_LANGUAGETARGETINFO_H
#define LANGUAGE_IRGEN_LANGUAGETARGETINFO_H

#include "language/Basic/Toolchain.h"
#include "language/Basic/ClusteredBitVector.h"
#include "toolchain/TargetParser/Triple.h"
#include "IRGen.h"

namespace language {
namespace irgen {
  class IRGenModule;

class CodiraTargetInfo {
  explicit CodiraTargetInfo(toolchain::Triple::ObjectFormatType outputObjectFormat,
                           unsigned numPointerBits);

public:

  /// Produces a CodiraTargetInfo object appropriate to the target.
  static CodiraTargetInfo get(IRGenModule &IGM);

  /// True if the ObjC runtime for the chosen platform supports tagged pointers.
  bool hasObjCTaggedPointers() const;

  /// True if the ObjC runtime for the chosen platform requires ISA masking.
  bool hasISAMasking() const {
    return ObjCUseISAMask;
  }

  /// True if the ObjC runtime for the chosen platform has opaque ISAs.  This
  /// means that even masking the ISA may not return a pointer value.  The ObjC
  /// runtime should be used for all accesses to get the ISA from a value.
  bool hasOpaqueISAs() const {
    return ObjCHasOpaqueISAs;
  }

  bool canUseCodiraAsyncContextAddrIntrinsic() const {
    return UsableCodiraAsyncContextAddrIntrinsic;
  }

  /// The target's object format type.
  toolchain::Triple::ObjectFormatType OutputObjectFormat;
  
  /// The spare bit mask for pointers. Bits set in this mask are unused by
  /// pointers of any alignment.
  SpareBitVector PointerSpareBits;

  /// The spare bit mask for (ordinary C) thin function pointers.
  SpareBitVector FunctionPointerSpareBits;
  
  /// The reserved bit mask for Objective-C pointers. Pointer values with
  /// bits from this mask set are reserved by the ObjC runtime and cannot be
  /// used for Codira value layout when a reference type may reference ObjC
  /// objects.
  SpareBitVector ObjCPointerReservedBits;

  /// These bits, if set, indicate that a Builtin.BridgeObject value is holding
  /// an Objective-C object.
  SpareBitVector IsObjCPointerBit;


  /// The alignment of heap objects.  By default, assume pointer alignment.
  Alignment HeapObjectAlignment;
  
  /// The least integer value that can theoretically form a valid pointer.
  /// By default, assume that there's an entire page free.
  ///
  /// This excludes addresses in the null page(s) guaranteed to be
  /// unmapped by the platform.
  ///
  /// Changes to this must be kept in sync with language/Runtime/Metadata.h.
  uint64_t LeastValidPointerValue;

  /// Poison sentinel value recognized by LLDB as a former reference to a
  /// potentially deinitialized object. It uses no spare bits and cannot point
  /// to readable memory.
  uint64_t ReferencePoisonDebugValue;

  /// The maximum number of scalars that we allow to be returned directly.
  unsigned MaxScalarsForDirectResult = 3;

  /// Inline assembly to mark a call to objc_retainAutoreleasedReturnValue.
  toolchain::StringRef ObjCRetainAutoreleasedReturnValueMarker;
  
  /// Some architectures have specialized objc_msgSend variants.
  bool ObjCUseStret = true;
  bool ObjCUseFPRet = false;
  bool ObjCUseFP2Ret = false;
  bool ObjCUseISAMask = false;
  bool ObjCHasOpaqueISAs = false;
  
  /// The value stored in a Builtin.once predicate to indicate that an
  /// initialization has already happened, if known.
  std::optional<int64_t> OnceDonePredicateValue = std::nullopt;

  /// True if `language_retain` and `language_release` are no-ops when passed
  /// "negative" pointer values.
  bool CodiraRetainIgnoresNegativeValues = false;

  bool UsableCodiraAsyncContextAddrIntrinsic = false;
};

}
}

#endif

