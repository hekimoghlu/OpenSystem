/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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

//===-- DiffLog.h - Difference Log Builder and accessories ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the interface to the LLVM difference log builder.
//
//===----------------------------------------------------------------------===//

#ifndef _LLVM_DIFFLOG_H_
#define _LLVM_DIFFLOG_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class Instruction;
  class Value;
  class Consumer;

  /// Trichotomy assumption
  enum DiffChange { DC_match, DC_left, DC_right };

  /// A temporary-object class for building up log messages.
  class LogBuilder {
    Consumer &consumer;

    /// The use of a stored StringRef here is okay because
    /// LogBuilder should be used only as a temporary, and as a
    /// temporary it will be destructed before whatever temporary
    /// might be initializing this format.
    StringRef Format;

    SmallVector<Value*, 4> Arguments;

  public:
    LogBuilder(Consumer &c, StringRef Format)
      : consumer(c), Format(Format) {}

    LogBuilder &operator<<(Value *V) {
      Arguments.push_back(V);
      return *this;
    }

    ~LogBuilder();

    StringRef getFormat() const;
    unsigned getNumArguments() const;
    Value *getArgument(unsigned I) const;
  };

  /// A temporary-object class for building up diff messages.
  class DiffLogBuilder {
    typedef std::pair<Instruction*,Instruction*> DiffRecord;
    SmallVector<DiffRecord, 20> Diff;

    Consumer &consumer;

  public:
    DiffLogBuilder(Consumer &c) : consumer(c) {}
    ~DiffLogBuilder();

    void addMatch(Instruction *L, Instruction *R);
    // HACK: VS 2010 has a bug in the stdlib that requires this.
    void addLeft(Instruction *L);
    void addRight(Instruction *R);

    unsigned getNumLines() const;
    DiffChange getLineKind(unsigned I) const;
    Instruction *getLeft(unsigned I) const;
    Instruction *getRight(unsigned I) const;
  };

}

#endif
