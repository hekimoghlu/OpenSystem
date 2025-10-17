/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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

//===- Main.cpp - Top-Level TableGen implementation -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TableGen is a tool which can be used to build up a description of something,
// then invoke one or more "tablegen backends" to emit information about the
// description in some predefined format.  In practice, this is used by the LLVM
// code generators to automate generation of a code generator through a
// high-level description of the target.
//
//===----------------------------------------------------------------------===//

#include "TGParser.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/system_error.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenAction.h"
#include <algorithm>
#include <cstdio>
using namespace llvm;

namespace {
  cl::opt<std::string>
  OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                 cl::init("-"));

  cl::opt<std::string>
  DependFilename("d",
                 cl::desc("Dependency filename"),
                 cl::value_desc("filename"),
                 cl::init(""));

  cl::opt<std::string>
  InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

  cl::list<std::string>
  IncludeDirs("I", cl::desc("Directory of include files"),
              cl::value_desc("directory"), cl::Prefix);
}

namespace llvm {

int TableGenMain(char *argv0, TableGenAction &Action) {
  RecordKeeper Records;

  try {
    // Parse the input file.
    OwningPtr<MemoryBuffer> File;
    if (error_code ec =
          MemoryBuffer::getFileOrSTDIN(InputFilename.c_str(), File)) {
      errs() << "Could not open input file '" << InputFilename << "': "
             << ec.message() <<"\n";
      return 1;
    }
    MemoryBuffer *F = File.take();

    // Tell SrcMgr about this buffer, which is what TGParser will pick up.
    SrcMgr.AddNewSourceBuffer(F, SMLoc());

    // Record the location of the include directory so that the lexer can find
    // it later.
    SrcMgr.setIncludeDirs(IncludeDirs);

    TGParser Parser(SrcMgr, Records);

    if (Parser.ParseFile())
      return 1;

    std::string Error;
    tool_output_file Out(OutputFilename.c_str(), Error);
    if (!Error.empty()) {
      errs() << argv0 << ": error opening " << OutputFilename
        << ":" << Error << "\n";
      return 1;
    }
    if (!DependFilename.empty()) {
      if (OutputFilename == "-") {
        errs() << argv0 << ": the option -d must be used together with -o\n";
        return 1;
      }
      tool_output_file DepOut(DependFilename.c_str(), Error);
      if (!Error.empty()) {
        errs() << argv0 << ": error opening " << DependFilename
          << ":" << Error << "\n";
        return 1;
      }
      DepOut.os() << OutputFilename << ":";
      const std::vector<std::string> &Dependencies = Parser.getDependencies();
      for (std::vector<std::string>::const_iterator I = Dependencies.begin(),
                                                    E = Dependencies.end();
           I != E; ++I) {
        DepOut.os() << " " << (*I);
      }
      DepOut.os() << "\n";
      DepOut.keep();
    }

    if (Action(Out.os(), Records))
      return 1;

    // Declare success.
    Out.keep();
    return 0;

  } catch (const TGError &Error) {
    PrintError(Error);
  } catch (const std::string &Error) {
    PrintError(Error);
  } catch (const char *Error) {
    PrintError(Error);
  } catch (...) {
    errs() << argv0 << ": Unknown unexpected exception occurred.\n";
  }

  return 1;
}

}
