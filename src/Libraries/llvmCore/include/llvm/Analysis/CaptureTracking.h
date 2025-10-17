/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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

//===----- llvm/Analysis/CaptureTracking.h - Pointer capture ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains routines that help determine which pointers are captured.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CAPTURETRACKING_H
#define LLVM_ANALYSIS_CAPTURETRACKING_H

#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Support/CallSite.h"

namespace llvm {
  /// PointerMayBeCaptured - Return true if this pointer value may be captured
  /// by the enclosing function (which is required to exist).  This routine can
  /// be expensive, so consider caching the results.  The boolean ReturnCaptures
  /// specifies whether returning the value (or part of it) from the function
  /// counts as capturing it or not.  The boolean StoreCaptures specified
  /// whether storing the value (or part of it) into memory anywhere
  /// automatically counts as capturing it or not.
  bool PointerMayBeCaptured(const Value *V,
                            bool ReturnCaptures,
                            bool StoreCaptures);

  /// This callback is used in conjunction with PointerMayBeCaptured. In
  /// addition to the interface here, you'll need to provide your own getters
  /// to see whether anything was captured.
  struct CaptureTracker {
    virtual ~CaptureTracker();

    /// tooManyUses - The depth of traversal has breached a limit. There may be
    /// capturing instructions that will not be passed into captured().
    virtual void tooManyUses() = 0;

    /// shouldExplore - This is the use of a value derived from the pointer.
    /// To prune the search (ie., assume that none of its users could possibly
    /// capture) return false. To search it, return true.
    ///
    /// U->getUser() is always an Instruction.
    virtual bool shouldExplore(Use *U) = 0;

    /// captured - Information about the pointer was captured by the user of
    /// use U. Return true to stop the traversal or false to continue looking
    /// for more capturing instructions.
    virtual bool captured(Use *U) = 0;
  };

  /// PointerMayBeCaptured - Visit the value and the values derived from it and
  /// find values which appear to be capturing the pointer value. This feeds
  /// results into and is controlled by the CaptureTracker object.
  void PointerMayBeCaptured(const Value *V, CaptureTracker *Tracker);
} // end namespace llvm

#endif
