/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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

//= llvm/Support/Win32/Mutex.inc - Win32 Reader/Writer Mutual Exclusion Lock  =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 specific (non-pthread) RWMutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

#include "Windows.h"

namespace llvm {
using namespace sys;

// Windows has slim read-writer lock support on Vista and higher, so we
// will attempt to load the APIs.  If they exist, we will use them, and
// if not, we will fall back on critical sections.  When we drop support
// for XP, we can stop lazy-loading these APIs and just use them directly.
#if defined(__MINGW32__)
  // Taken from WinNT.h
  typedef struct _RTL_SRWLOCK {
    PVOID Ptr;
  } RTL_SRWLOCK, *PRTL_SRWLOCK;

  // Taken from WinBase.h
  typedef RTL_SRWLOCK SRWLOCK, *PSRWLOCK;
#endif

static VOID (WINAPI *fpInitializeSRWLock)(PSRWLOCK lock) = NULL;
static VOID (WINAPI *fpAcquireSRWLockExclusive)(PSRWLOCK lock) = NULL;
static VOID (WINAPI *fpAcquireSRWLockShared)(PSRWLOCK lock) = NULL;
static VOID (WINAPI *fpReleaseSRWLockExclusive)(PSRWLOCK lock) = NULL;
static VOID (WINAPI *fpReleaseSRWLockShared)(PSRWLOCK lock) = NULL;

static bool sHasSRW = false;

static bool loadSRW() {
  static bool sChecked = false;
  if (!sChecked) {
    sChecked = true;

    HMODULE hLib = ::LoadLibrary(TEXT("Kernel32"));
    if (hLib) {
      fpInitializeSRWLock =
        (VOID (WINAPI *)(PSRWLOCK))::GetProcAddress(hLib,
                                               "InitializeSRWLock");
      fpAcquireSRWLockExclusive =
        (VOID (WINAPI *)(PSRWLOCK))::GetProcAddress(hLib,
                                               "AcquireSRWLockExclusive");
      fpAcquireSRWLockShared =
        (VOID (WINAPI *)(PSRWLOCK))::GetProcAddress(hLib,
                                               "AcquireSRWLockShared");
      fpReleaseSRWLockExclusive =
        (VOID (WINAPI *)(PSRWLOCK))::GetProcAddress(hLib,
                                               "ReleaseSRWLockExclusive");
      fpReleaseSRWLockShared =
        (VOID (WINAPI *)(PSRWLOCK))::GetProcAddress(hLib,
                                               "ReleaseSRWLockShared");
      ::FreeLibrary(hLib);

      if (fpInitializeSRWLock != NULL) {
        sHasSRW = true;
      }
    }
  }
  return sHasSRW;
}

RWMutexImpl::RWMutexImpl() {
  if (loadSRW()) {
    data_ = calloc(1, sizeof(SRWLOCK));
    fpInitializeSRWLock(static_cast<PSRWLOCK>(data_));
  } else {
    data_ = calloc(1, sizeof(CRITICAL_SECTION));
    InitializeCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  }
}

RWMutexImpl::~RWMutexImpl() {
  if (sHasSRW) {
    // Nothing to do in the case of slim reader/writers
  } else {
    DeleteCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
    free(data_);
  }
}

bool RWMutexImpl::reader_acquire() {
  if (sHasSRW) {
    fpAcquireSRWLockShared(static_cast<PSRWLOCK>(data_));
  } else {
    EnterCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  }
  return true;
}

bool RWMutexImpl::reader_release() {
  if (sHasSRW) {
    fpReleaseSRWLockShared(static_cast<PSRWLOCK>(data_));
  } else {
    LeaveCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  }
  return true;
}

bool RWMutexImpl::writer_acquire() {
  if (sHasSRW) {
    fpAcquireSRWLockExclusive(static_cast<PSRWLOCK>(data_));
  } else {
    EnterCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  }
  return true;
}

bool RWMutexImpl::writer_release() {
  if (sHasSRW) {
    fpReleaseSRWLockExclusive(static_cast<PSRWLOCK>(data_));
  } else {
    LeaveCriticalSection(static_cast<LPCRITICAL_SECTION>(data_));
  }
  return true;
}


}
