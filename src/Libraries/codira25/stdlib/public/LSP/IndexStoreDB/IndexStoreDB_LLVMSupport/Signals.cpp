/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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

//===- Signals.cpp - Signal Handling support --------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occurring while your program is running.
//
//===----------------------------------------------------------------------===//

#include <IndexStoreDB_LLVMSupport/toolchain_Support_Signals.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_STLExtras.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Config_toolchain-config.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_ErrorOr.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_FileSystem.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_FileUtilities.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Format.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_FormatVariadic.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_FormatAdapters.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_ManagedStatic.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_MemoryBuffer.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Mutex.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Program.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_StringSaver.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_raw_ostream.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Options.h>
#include <vector>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

using namespace toolchain;

// Use explicit storage to avoid accessing cl::opt in a signal handler.
static bool DisableSymbolicationFlag = false;
static cl::opt<bool, true>
    DisableSymbolication("disable-symbolication",
                         cl::desc("Disable symbolizing crash backtraces."),
                         cl::location(DisableSymbolicationFlag), cl::Hidden);

// Callbacks to run in signal handler must be lock-free because a signal handler
// could be running as we add new callbacks. We don't add unbounded numbers of
// callbacks, an array is therefore sufficient.
struct CallbackAndCookie {
  sys::SignalHandlerCallback Callback;
  void *Cookie;
  enum class Status { Empty, Initializing, Initialized, Executing };
  std::atomic<Status> Flag;
};
static constexpr size_t MaxSignalHandlerCallbacks = 8;
static CallbackAndCookie CallBacksToRun[MaxSignalHandlerCallbacks];

// Signal-safe.
void sys::RunSignalHandlers() {
  for (size_t I = 0; I < MaxSignalHandlerCallbacks; ++I) {
    auto &RunMe = CallBacksToRun[I];
    auto Expected = CallbackAndCookie::Status::Initialized;
    auto Desired = CallbackAndCookie::Status::Executing;
    if (!RunMe.Flag.compare_exchange_strong(Expected, Desired))
      continue;
    (*RunMe.Callback)(RunMe.Cookie);
    RunMe.Callback = nullptr;
    RunMe.Cookie = nullptr;
    RunMe.Flag.store(CallbackAndCookie::Status::Empty);
  }
}

// Signal-safe.
static void insertSignalHandler(sys::SignalHandlerCallback FnPtr,
                                void *Cookie) {
  for (size_t I = 0; I < MaxSignalHandlerCallbacks; ++I) {
    auto &SetMe = CallBacksToRun[I];
    auto Expected = CallbackAndCookie::Status::Empty;
    auto Desired = CallbackAndCookie::Status::Initializing;
    if (!SetMe.Flag.compare_exchange_strong(Expected, Desired))
      continue;
    SetMe.Callback = FnPtr;
    SetMe.Cookie = Cookie;
    SetMe.Flag.store(CallbackAndCookie::Status::Initialized);
    return;
  }
  report_fatal_error("too many signal callbacks already registered");
}

static bool findModulesAndOffsets(void **StackTrace, int Depth,
                                  const char **Modules, intptr_t *Offsets,
                                  const char *MainExecutableName,
                                  StringSaver &StrPool);

/// Format a pointer value as hexadecimal. Zero pad it out so its always the
/// same width.
static FormattedNumber format_ptr(void *PC) {
  // Each byte is two hex digits plus 2 for the 0x prefix.
  unsigned PtrWidth = 2 + 2 * sizeof(void *);
  return format_hex((uint64_t)PC, PtrWidth);
}

/// Helper that launches toolchain-symbolizer and symbolizes a backtrace.
LLVM_ATTRIBUTE_USED
static bool printSymbolizedStackTrace(StringRef Argv0, void **StackTrace,
                                      int Depth, toolchain::raw_ostream &OS) {
  if (DisableSymbolicationFlag)
    return false;

  // Don't recursively invoke the toolchain-symbolizer binary.
  if (Argv0.find("toolchain-symbolizer") != std::string::npos)
    return false;

  // FIXME: Subtract necessary number from StackTrace entries to turn return addresses
  // into actual instruction addresses.
  // Use toolchain-symbolizer tool to symbolize the stack traces. First look for it
  // alongside our binary, then in $PATH.
  ErrorOr<std::string> LLVMSymbolizerPathOrErr = std::error_code();
  if (!Argv0.empty()) {
    StringRef Parent = toolchain::sys::path::parent_path(Argv0);
    if (!Parent.empty())
      LLVMSymbolizerPathOrErr = sys::findProgramByName("toolchain-symbolizer", Parent);
  }
  if (!LLVMSymbolizerPathOrErr)
    LLVMSymbolizerPathOrErr = sys::findProgramByName("toolchain-symbolizer");
  if (!LLVMSymbolizerPathOrErr)
    return false;
  const std::string &LLVMSymbolizerPath = *LLVMSymbolizerPathOrErr;

  // If we don't know argv0 or the address of main() at this point, try
  // to guess it anyway (it's possible on some platforms).
  std::string MainExecutableName =
      sys::fs::exists(Argv0) ? (std::string)Argv0
                             : sys::fs::getMainExecutable(nullptr, nullptr);
  BumpPtrAllocator Allocator;
  StringSaver StrPool(Allocator);
  std::vector<const char *> Modules(Depth, nullptr);
  std::vector<intptr_t> Offsets(Depth, 0);
  if (!findModulesAndOffsets(StackTrace, Depth, Modules.data(), Offsets.data(),
                             MainExecutableName.c_str(), StrPool))
    return false;
  int InputFD;
  SmallString<32> InputFile, OutputFile;
  sys::fs::createTemporaryFile("symbolizer-input", "", InputFD, InputFile);
  sys::fs::createTemporaryFile("symbolizer-output", "", OutputFile);
  FileRemover InputRemover(InputFile.c_str());
  FileRemover OutputRemover(OutputFile.c_str());

  {
    raw_fd_ostream Input(InputFD, true);
    for (int i = 0; i < Depth; i++) {
      if (Modules[i])
        Input << Modules[i] << " " << (void*)Offsets[i] << "\n";
    }
  }

  Optional<StringRef> Redirects[] = {StringRef(InputFile),
                                     StringRef(OutputFile), StringRef("")};
  StringRef Args[] = {"toolchain-symbolizer", "--functions=linkage", "--inlining",
#ifdef _WIN32
                      // Pass --relative-address on Windows so that we don't
                      // have to add ImageBase from PE file.
                      // FIXME: Make this the default for toolchain-symbolizer.
                      "--relative-address",
#endif
                      "--demangle"};
  int RunResult =
      sys::ExecuteAndWait(LLVMSymbolizerPath, Args, None, Redirects);
  if (RunResult != 0)
    return false;

  // This report format is based on the sanitizer stack trace printer.  See
  // sanitizer_stacktrace_printer.cc in compiler-rt.
  auto OutputBuf = MemoryBuffer::getFile(OutputFile.c_str());
  if (!OutputBuf)
    return false;
  StringRef Output = OutputBuf.get()->getBuffer();
  SmallVector<StringRef, 32> Lines;
  Output.split(Lines, "\n");
  auto CurLine = Lines.begin();
  int frame_no = 0;
  for (int i = 0; i < Depth; i++) {
    auto PrintLineHeader = [&]() {
      OS << right_justify(formatv("#{0}", frame_no++).str(),
                          std::log10(Depth) + 2)
         << ' ' << format_ptr(StackTrace[i]) << ' ';
    };
    if (!Modules[i]) {
      PrintLineHeader();
      OS << '\n';
      continue;
    }
    // Read pairs of lines (function name and file/line info) until we
    // encounter empty line.
    for (;;) {
      if (CurLine == Lines.end())
        return false;
      StringRef FunctionName = *CurLine++;
      if (FunctionName.empty())
        break;
      PrintLineHeader();
      if (!FunctionName.startswith("??"))
        OS << FunctionName << ' ';
      if (CurLine == Lines.end())
        return false;
      StringRef FileLineInfo = *CurLine++;
      if (!FileLineInfo.startswith("??"))
        OS << FileLineInfo;
      else
        OS << "(" << Modules[i] << '+' << format_hex(Offsets[i], 0) << ")";
      OS << "\n";
    }
  }
  return true;
}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Signals.inc"
#endif
#ifdef _WIN32
#include "Windows/Signals.inc"
#endif
