/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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

//===-- DifferenceEngine.h - Module comparator ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the interface to the LLVM difference engine,
// which structurally compares functions within a module.
//
//===----------------------------------------------------------------------===//

#ifndef _LLVM_DIFFERENCE_ENGINE_H_
#define _LLVM_DIFFERENCE_ENGINE_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "DiffLog.h"
#include "DiffConsumer.h"

#include <utility>

namespace llvm {
  class Function;
  class GlobalValue;
  class Instruction;
  class LLVMContext;
  class Module;
  class Twine;
  class Value;

  /// A class for performing structural comparisons of LLVM assembly.
  class DifferenceEngine {
  public:
    /// A RAII object for recording the current context.
    struct Context {
      Context(DifferenceEngine &Engine, Value *L, Value *R) : Engine(Engine) {
        Engine.consumer.enterContext(L, R);
      }

      ~Context() {
        Engine.consumer.exitContext();
      }

    private:
      DifferenceEngine &Engine;
    };

    /// An oracle for answering whether two values are equivalent as
    /// operands.
    class Oracle {
      virtual void anchor();
    public:
      virtual bool operator()(Value *L, Value *R) = 0;

    protected:
      virtual ~Oracle() {}
    };

    DifferenceEngine(Consumer &consumer)
      : consumer(consumer), globalValueOracle(0) {}

    void diff(Module *L, Module *R);
    void diff(Function *L, Function *R);
    void log(StringRef text) {
      consumer.log(text);
    }
    LogBuilder logf(StringRef text) {
      return LogBuilder(consumer, text);
    }
    Consumer& getConsumer() const { return consumer; }

    /// Installs an oracle to decide whether two global values are
    /// equivalent as operands.  Without an oracle, global values are
    /// considered equivalent as operands precisely when they have the
    /// same name.
    void setGlobalValueOracle(Oracle *oracle) {
      globalValueOracle = oracle;
    }

    /// Determines whether two global values are equivalent.
    bool equivalentAsOperands(GlobalValue *L, GlobalValue *R);

  private:
    Consumer &consumer;
    Oracle *globalValueOracle;
  };
}

#endif
