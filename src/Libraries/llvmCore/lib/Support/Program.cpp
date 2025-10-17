/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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

//===-- Program.cpp - Implement OS Program Concept --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Program concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Program.h"
#include "llvm/Config/config.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

int
Program::ExecuteAndWait(const Path& path,
                        const char** args,
                        const char** envp,
                        const Path** redirects,
                        unsigned secondsToWait,
                        unsigned memoryLimit,
                        std::string* ErrMsg) {
  Program prg;
  if (prg.Execute(path, args, envp, redirects, memoryLimit, ErrMsg))
    return prg.Wait(path, secondsToWait, ErrMsg);
  else
    return -1;
}

void
Program::ExecuteNoWait(const Path& path,
                       const char** args,
                       const char** envp,
                       const Path** redirects,
                       unsigned memoryLimit,
                       std::string* ErrMsg) {
  Program prg;
  prg.Execute(path, args, envp, redirects, memoryLimit, ErrMsg);
}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Program.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Program.inc"
#endif
