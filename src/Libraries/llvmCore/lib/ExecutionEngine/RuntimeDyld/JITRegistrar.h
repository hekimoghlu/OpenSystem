/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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

//===-- JITRegistrar.h - Registers objects with a debugger ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTION_ENGINE_JIT_REGISTRAR_H
#define LLVM_EXECUTION_ENGINE_JIT_REGISTRAR_H

#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

/// Global access point for the JIT debugging interface.
class JITRegistrar {
public:
  /// Instantiates the JIT service.
  JITRegistrar() {}

  /// Unregisters each object that was previously registered and releases all
  /// internal resources.
  virtual ~JITRegistrar() {}

  /// Creates an entry in the JIT registry for the buffer @p Object,
  /// which must contain an object file in executable memory with any
  /// debug information for the debugger.
  virtual void registerObject(const MemoryBuffer &Object) = 0;

  /// Removes the internal registration of @p Object, and
  /// frees associated resources.
  /// Returns true if @p Object was previously registered.
  virtual bool deregisterObject(const MemoryBuffer &Object) = 0;

  /// Returns a reference to a GDB JIT registrar singleton
  static JITRegistrar& getGDBRegistrar();
};

} // end namespace llvm

#endif // LLVM_EXECUTION_ENGINE_JIT_REGISTRAR_H
