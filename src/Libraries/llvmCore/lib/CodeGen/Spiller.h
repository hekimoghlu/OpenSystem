/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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

//===-- llvm/CodeGen/Spiller.h - Spiller -*- C++ -*------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPILLER_H
#define LLVM_CODEGEN_SPILLER_H

namespace llvm {

  class LiveRangeEdit;
  class MachineFunction;
  class MachineFunctionPass;
  class VirtRegMap;

  /// Spiller interface.
  ///
  /// Implementations are utility classes which insert spill or remat code on
  /// demand.
  class Spiller {
    virtual void anchor();
  public:
    virtual ~Spiller() = 0;

    /// spill - Spill the LRE.getParent() live interval.
    virtual void spill(LiveRangeEdit &LRE) = 0;

  };

  /// Create and return a spiller object, as specified on the command line.
  Spiller* createSpiller(MachineFunctionPass &pass,
                         MachineFunction &mf,
                         VirtRegMap &vrm);

  /// Create and return a spiller that will insert spill code directly instead
  /// of deferring though VirtRegMap.
  Spiller *createInlineSpiller(MachineFunctionPass &pass,
                               MachineFunction &mf,
                               VirtRegMap &vrm);

}

#endif
