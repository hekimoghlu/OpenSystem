/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

//===- PrettyStackTrace.cpp - Pretty Crash Handling -----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occurring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"     // Get autoconf configuration settings
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadLocal.h"
#include "llvm/ADT/SmallString.h"

#ifdef HAVE_CRASHREPORTERCLIENT_H
#include <CrashReporterClient.h>
#endif

using namespace llvm;

namespace llvm {
  bool DisablePrettyStackTrace = false;
}

// FIXME: This should be thread local when llvm supports threads.
static sys::ThreadLocal<const PrettyStackTraceEntry> PrettyStackTraceHead;

static unsigned PrintStack(const PrettyStackTraceEntry *Entry, raw_ostream &OS){
  unsigned NextID = 0;
  if (Entry->getNextEntry())
    NextID = PrintStack(Entry->getNextEntry(), OS);
  OS << NextID << ".\t";
  Entry->print(OS);
  
  return NextID+1;
}

/// PrintCurStackTrace - Print the current stack trace to the specified stream.
static void PrintCurStackTrace(raw_ostream &OS) {
  // Don't print an empty trace.
  if (PrettyStackTraceHead.get() == 0) return;
  
  // If there are pretty stack frames registered, walk and emit them.
  OS << "Stack dump:\n";
  
  PrintStack(PrettyStackTraceHead.get(), OS);
  OS.flush();
}

// Integrate with crash reporter libraries.
#if defined (__APPLE__) && HAVE_CRASHREPORTERCLIENT_H
//  If any clients of llvm try to link to libCrashReporterClient.a themselves,
//  only one crash info struct will be used.
extern "C" {
CRASH_REPORTER_CLIENT_HIDDEN 
struct crashreporter_annotations_t gCRAnnotations 
        __attribute__((section("__DATA," CRASHREPORTER_ANNOTATIONS_SECTION))) 
        = { CRASHREPORTER_ANNOTATIONS_VERSION, 0, 0, 0, 0, 0, 0 };
}
#elif defined (__APPLE__) && HAVE_CRASHREPORTER_INFO
static const char *__crashreporter_info__ = 0;
asm(".desc ___crashreporter_info__, 0x10");
#endif


/// CrashHandler - This callback is run if a fatal signal is delivered to the
/// process, it prints the pretty stack trace.
static void CrashHandler(void *) {
#ifndef __APPLE__
  // On non-apple systems, just emit the crash stack trace to stderr.
  PrintCurStackTrace(errs());
#else
  // Otherwise, emit to a smallvector of chars, send *that* to stderr, but also
  // put it into __crashreporter_info__.
  SmallString<2048> TmpStr;
  {
    raw_svector_ostream Stream(TmpStr);
    PrintCurStackTrace(Stream);
  }
  
  if (!TmpStr.empty()) {
#ifdef HAVE_CRASHREPORTERCLIENT_H
    // Cast to void to avoid warning.
    (void)CRSetCrashLogMessage(std::string(TmpStr.str()).c_str());
#elif HAVE_CRASHREPORTER_INFO 
    __crashreporter_info__ = strdup(std::string(TmpStr.str()).c_str());
#endif
    errs() << TmpStr.str();
  }
  
#endif
}

static bool RegisterCrashPrinter() {
  if (!DisablePrettyStackTrace)
    sys::AddSignalHandler(CrashHandler, 0);
  return false;
}

PrettyStackTraceEntry::PrettyStackTraceEntry() {
  // The first time this is called, we register the crash printer.
  static bool HandlerRegistered = RegisterCrashPrinter();
  (void)HandlerRegistered;
    
  // Link ourselves.
  NextEntry = PrettyStackTraceHead.get();
  PrettyStackTraceHead.set(this);
}

PrettyStackTraceEntry::~PrettyStackTraceEntry() {
  assert(PrettyStackTraceHead.get() == this &&
         "Pretty stack trace entry destruction is out of order");
  PrettyStackTraceHead.set(getNextEntry());
}

void PrettyStackTraceString::print(raw_ostream &OS) const {
  OS << Str << "\n";
}

void PrettyStackTraceProgram::print(raw_ostream &OS) const {
  OS << "Program arguments: ";
  // Print the argument list.
  for (unsigned i = 0, e = ArgC; i != e; ++i)
    OS << ArgV[i] << ' ';
  OS << '\n';
}
