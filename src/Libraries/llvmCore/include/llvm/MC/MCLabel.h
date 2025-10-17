/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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

//===- MCLabel.h - Machine Code Directional Local Labels --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCLabel class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCLABEL_H
#define LLVM_MC_MCLABEL_H

#include "llvm/Support/Compiler.h"

namespace llvm {
  class MCContext;
  class raw_ostream;

  /// MCLabel - Instances of this class represent a label name in the MC file,
  /// and MCLabel are created and unique'd by the MCContext class.  MCLabel
  /// should only be constructed for valid instances in the object file.
  class MCLabel {
    // Instance - the instance number of this Directional Local Label
    unsigned Instance;

  private:  // MCContext creates and uniques these.
    friend class MCContext;
    MCLabel(unsigned instance)
      : Instance(instance) {}

    MCLabel(const MCLabel&) LLVM_DELETED_FUNCTION;
    void operator=(const MCLabel&) LLVM_DELETED_FUNCTION;
  public:
    /// getInstance - Get the current instance of this Directional Local Label.
    unsigned getInstance() const { return Instance; }

    /// incInstance - Increment the current instance of this Directional Local
    /// Label.
    unsigned incInstance() { return ++Instance; }

    /// print - Print the value to the stream \p OS.
    void print(raw_ostream &OS) const;

    /// dump - Print the value to stderr.
    void dump() const;
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const MCLabel &Label) {
    Label.print(OS);
    return OS;
  }
} // end namespace llvm

#endif
