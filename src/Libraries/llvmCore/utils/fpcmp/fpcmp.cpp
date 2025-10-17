/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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

//===- fpcmp.cpp - A fuzzy "cmp" that permits floating point noise --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// fpcmp is a tool that basically works like the 'cmp' tool, except that it can
// tolerate errors due to floating point noise, with the -r and -a options.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
  cl::opt<std::string>
  File1(cl::Positional, cl::desc("<input file #1>"), cl::Required);
  cl::opt<std::string>
  File2(cl::Positional, cl::desc("<input file #2>"), cl::Required);

  cl::opt<double>
  RelTolerance("r", cl::desc("Relative error tolerated"), cl::init(0));
  cl::opt<double>
  AbsTolerance("a", cl::desc("Absolute error tolerated"), cl::init(0));
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  std::string ErrorMsg;
  int DF = DiffFilesWithTolerance(sys::PathWithStatus(File1), 
                                  sys::PathWithStatus(File2),
                                  AbsTolerance, RelTolerance, &ErrorMsg);
  if (!ErrorMsg.empty())
    errs() << argv[0] << ": " << ErrorMsg << "\n";
  return DF;
}

