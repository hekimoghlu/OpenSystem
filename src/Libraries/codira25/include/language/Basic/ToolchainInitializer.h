/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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

//===--- ToolchainInitializer.h ---------------------------------------*- C++ -*-===//
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
// A file that declares macros for initializing all parts of LLVM that various
// binaries in language use. Please call PROGRAM_START in the main routine of all
// binaries, and INITIALIZE_LLVM in anything that uses Clang or LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_LLVMINITIALIZE_H
#define LANGUAGE_BASIC_LLVMINITIALIZE_H

#include "toolchain/Support/InitToolchain.h"
#include "toolchain/Support/ManagedStatic.h"
#include "toolchain/Support/PrettyStackTrace.h"
#include "toolchain/Support/Signals.h"
#include "toolchain/Support/TargetSelect.h"

#define PROGRAM_START(argc, argv) \
  toolchain::InitLLVM _INITIALIZE_LLVM(argc, argv)

#define INITIALIZE_LLVM() \
  do { \
    toolchain::InitializeAllTargets(); \
    toolchain::InitializeAllTargetMCs(); \
    toolchain::InitializeAllAsmPrinters(); \
    toolchain::InitializeAllAsmParsers(); \
    toolchain::InitializeAllDisassemblers(); \
    toolchain::InitializeAllTargetInfos(); \
  } while (0)

#endif // LANGUAGE_BASIC_LLVMINITIALIZE_H
