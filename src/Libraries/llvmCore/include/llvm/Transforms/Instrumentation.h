/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

//===- Transforms/Instrumentation.h - Instrumentation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines constructor functions for instrumentation passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_H

namespace llvm {

class ModulePass;
class FunctionPass;

// Insert edge profiling instrumentation
ModulePass *createEdgeProfilerPass();

// Insert optimal edge profiling instrumentation
ModulePass *createOptimalEdgeProfilerPass();

// Insert path profiling instrumentation
ModulePass *createPathProfilerPass();

// Insert GCOV profiling instrumentation
ModulePass *createGCOVProfilerPass(bool EmitNotes = true, bool EmitData = true,
                                   bool Use402Format = false,
                                   bool UseExtraChecksum = false);

// Insert AddressSanitizer (address sanity checking) instrumentation
ModulePass *createAddressSanitizerPass();
// Insert ThreadSanitizer (race detection) instrumentation
FunctionPass *createThreadSanitizerPass();


// BoundsChecking - This pass instruments the code to perform run-time bounds
// checking on loads, stores, and other memory intrinsics.
// Penalty is the maximum run-time that is acceptable for the user.
//
FunctionPass *createBoundsCheckingPass(unsigned Penalty = 5);

} // End llvm namespace

#endif
