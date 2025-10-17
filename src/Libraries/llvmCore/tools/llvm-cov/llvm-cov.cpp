/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

//===- tools/llvm-cov/llvm-cov.cpp - LLVM coverage tool -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// llvm-cov is a command line tools to analyze and report coverage information.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GCOV.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

static cl::opt<bool>
DumpGCOV("dump", cl::init(false), cl::desc("dump gcov file"));

static cl::opt<std::string>
InputGCNO("gcno", cl::desc("<input gcno file>"), cl::init(""));

static cl::opt<std::string>
InputGCDA("gcda", cl::desc("<input gcda file>"), cl::init(""));


//===----------------------------------------------------------------------===//
int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm cov\n");

  GCOVFile GF;
  if (InputGCNO.empty())
    errs() << " " << argv[0] << ": No gcov input file!\n";

  OwningPtr<MemoryBuffer> GCNO_Buff;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCNO, GCNO_Buff)) {
    errs() << InputGCNO << ": " << ec.message() << "\n";
    return 1;
  }
  GCOVBuffer GCNO_GB(GCNO_Buff.take());
  if (!GF.read(GCNO_GB)) {
    errs() << "Invalid .gcno File!\n";
    return 1;
  }

  if (!InputGCDA.empty()) {
    OwningPtr<MemoryBuffer> GCDA_Buff;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputGCDA, GCDA_Buff)) {
      errs() << InputGCDA << ": " << ec.message() << "\n";
      return 1;
    }
    GCOVBuffer GCDA_GB(GCDA_Buff.take());
    if (!GF.read(GCDA_GB)) {
      errs() << "Invalid .gcda File!\n";
      return 1;
    }
  }


  if (DumpGCOV)
    GF.dump();

  FileInfo FI;
  GF.collectLineCounts(FI);
  return 0;
}
