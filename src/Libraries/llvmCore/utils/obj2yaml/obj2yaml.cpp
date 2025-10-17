/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

//===------ utils/obj2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"

#include "llvm/ADT/OwningPtr.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"

const char endl = '\n';

namespace yaml {  // generic yaml-writing specific routines

unsigned char printable(unsigned char Ch) {
  return Ch >= ' ' && Ch <= '~' ? Ch : '.';
}
  
llvm::raw_ostream &writeHexStream(llvm::raw_ostream &Out, 
                                     const llvm::ArrayRef<uint8_t> arr) {
  const char *hex = "0123456789ABCDEF";
  Out << " !hex \"";

  typedef llvm::ArrayRef<uint8_t>::const_iterator iter_t;
  const iter_t end = arr.end();
  for (iter_t iter = arr.begin(); iter != end; ++iter)
    Out << hex[(*iter >> 4) & 0x0F] << hex[(*iter & 0x0F)];

  Out << "\" # |";
  for (iter_t iter = arr.begin(); iter != end; ++iter)
    Out << printable(*iter);
  Out << "|" << endl;

  return Out;
  }

llvm::raw_ostream &writeHexNumber(llvm::raw_ostream &Out, unsigned long long N) {
  if (N >= 10)
    Out << "0x";
  Out.write_hex(N);
  return Out;
}

}


using namespace llvm;
enum ObjectFileType { coff };

cl::opt<ObjectFileType> InputFormat(
  cl::desc("Choose input format"),
    cl::values(
      clEnumVal(coff, "process COFF object files"),
    clEnumValEnd));
    
cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

int main(int argc, char * argv[]) {
  cl::ParseCommandLineOptions(argc, argv);
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

// Process the input file  
  OwningPtr<MemoryBuffer> buf;

// TODO: If this is an archive, then burst it and dump each entry
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFilename, buf))
    llvm::errs() << "Error: '" << ec.message() << "' opening file '" 
              << InputFilename << "'" << endl;
  else {
    ec = coff2yaml(llvm::outs(), buf.take());
    if (ec)
      llvm::errs() << "Error: " << ec.message() << " dumping COFF file" << endl;
  }

  return 0;
}
