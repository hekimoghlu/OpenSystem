/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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

//===-- OcamlGC.cpp - Ocaml frametable GC strategy ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering for the llvm.gc* intrinsics compatible with
// Objective Caml 3.10.0, which uses a liveness-accurate static stack map.
//
// The frametable emitter is in OcamlGCPrinter.cpp.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/GCStrategy.h"

using namespace llvm;

namespace {
  class OcamlGC : public GCStrategy {
  public:
    OcamlGC();
  };
}

static GCRegistry::Add<OcamlGC>
X("ocaml", "ocaml 3.10-compatible GC");

void llvm::linkOcamlGC() { }

OcamlGC::OcamlGC() {
  NeededSafePoints = 1 << GC::PostCall;
  UsesMetadata = true;
}
