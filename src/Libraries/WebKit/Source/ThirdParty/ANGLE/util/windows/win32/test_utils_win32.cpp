/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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

//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// test_utils_win32.cpp: Implementation of OS-specific functions for Win32 (Windows)

#include "util/test_utils.h"

#include <windows.h>
#include <array>

#include "util/windows/third_party/StackWalker/src/StackWalker.h"

namespace angle
{
namespace
{
static const struct
{
    const char *name;
    const DWORD code;
} kExceptions[] = {
#define _(E)  \
    {         \
#        E, E \
    }
    _(EXCEPTION_ACCESS_VIOLATION),
    _(EXCEPTION_BREAKPOINT),
    _(EXCEPTION_INT_DIVIDE_BY_ZERO),
    _(EXCEPTION_STACK_OVERFLOW),
#undef _
};

class CustomStackWalker : public StackWalker
{
  public:
    CustomStackWalker() {}
    ~CustomStackWalker() override {}

    void OnCallstackEntry(CallstackEntryType eType, CallstackEntry &entry) override
    {
        char buffer[STACKWALK_MAX_NAMELEN];
        size_t maxLen = _TRUNCATE;
        if ((eType != lastEntry) && (entry.offset != 0))
        {
            if (entry.name[0] == 0)
                strncpy_s(entry.name, STACKWALK_MAX_NAMELEN, "(function-name not available)",
                          _TRUNCATE);
            if (entry.undName[0] != 0)
                strncpy_s(entry.name, STACKWALK_MAX_NAMELEN, entry.undName, _TRUNCATE);
            if (entry.undFullName[0] != 0)
                strncpy_s(entry.name, STACKWALK_MAX_NAMELEN, entry.undFullName, _TRUNCATE);
            if (entry.lineFileName[0] == 0)
            {
                strncpy_s(entry.lineFileName, STACKWALK_MAX_NAMELEN, "(filename not available)",
                          _TRUNCATE);
                if (entry.moduleName[0] == 0)
                    strncpy_s(entry.moduleName, STACKWALK_MAX_NAMELEN,
                              "(module-name not available)", _TRUNCATE);
                _snprintf_s(buffer, maxLen, "    %s - %p (%s): %s\n", entry.name,
                            reinterpret_cast<void *>(entry.offset), entry.moduleName,
                            entry.lineFileName);
            }
            else
                _snprintf_s(buffer, maxLen, "    %s (%s:%d)\n", entry.name, entry.lineFileName,
                            entry.lineNumber);
            buffer[STACKWALK_MAX_NAMELEN - 1] = 0;
            printf("%s", buffer);
            OutputDebugStringA(buffer);
        }
    }
};

void PrintBacktrace(CONTEXT *c)
{
    printf("Backtrace:\n");
    OutputDebugStringA("Backtrace:\n");

    CustomStackWalker sw;
    sw.ShowCallstack(GetCurrentThread(), c);
}

LONG WINAPI StackTraceCrashHandler(EXCEPTION_POINTERS *e)
{
    const DWORD code = e->ExceptionRecord->ExceptionCode;
    printf("\nCaught exception %lu", code);
    for (size_t i = 0; i < ArraySize(kExceptions); i++)
    {
        if (kExceptions[i].code == code)
        {
            printf(" %s", kExceptions[i].name);
        }
    }
    printf("\n");

    PrintBacktrace(e->ContextRecord);

    // Exit NOW.  Don't notify other threads, don't call anything registered with atexit().
    _exit(1);

    // The compiler wants us to return something.  This is what we'd do if we didn't _exit().
    return EXCEPTION_EXECUTE_HANDLER;
}

CrashCallback *gCrashHandlerCallback;

LONG WINAPI CrashHandler(EXCEPTION_POINTERS *e)
{
    if (gCrashHandlerCallback)
    {
        (*gCrashHandlerCallback)();
    }
    return StackTraceCrashHandler(e);
}
}  // namespace

void SetLowPriorityProcess()
{
    ::SetPriorityClass(::GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
}

bool StabilizeCPUForBenchmarking()
{
    if (::SetThreadAffinityMask(::GetCurrentThread(), 1) == 0)
    {
        return false;
    }
    if (::SetPriorityClass(::GetCurrentProcess(), REALTIME_PRIORITY_CLASS) == FALSE)
    {
        return false;
    }
    if (::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) == FALSE)
    {
        return false;
    }

    return true;
}

void PrintStackBacktrace()
{
    CONTEXT context;
    ZeroMemory(&context, sizeof(CONTEXT));
    RtlCaptureContext(&context);
    PrintBacktrace(&context);
}

void InitCrashHandler(CrashCallback *callback)
{
    if (callback)
    {
        gCrashHandlerCallback = callback;
    }
    SetUnhandledExceptionFilter(CrashHandler);
}

void TerminateCrashHandler()
{
    gCrashHandlerCallback = nullptr;
    SetUnhandledExceptionFilter(nullptr);
}

int NumberOfProcessors()
{
    // A portable implementation could probably use GetLogicalProcessorInformation
    return ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
}
}  // namespace angle
