/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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

//= llvm/Support/Win32/ThreadLocal.inc - Win32 Thread Local Data -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 specific (non-pthread) ThreadLocal class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Win32 code that
//===          is guaranteed to work on *all* Win32 variants.
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include "llvm/Support/ThreadLocal.h"

namespace llvm {
using namespace sys;

ThreadLocalImpl::ThreadLocalImpl() : data() {
  typedef int SIZE_TOO_BIG[sizeof(DWORD) <= sizeof(data) ? 1 : -1];
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  *tls = TlsAlloc();
  assert(*tls != TLS_OUT_OF_INDEXES);
}

ThreadLocalImpl::~ThreadLocalImpl() {
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  TlsFree(*tls);
}

const void* ThreadLocalImpl::getInstance() {
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  return TlsGetValue(*tls);
}

void ThreadLocalImpl::setInstance(const void* d){
  DWORD* tls = reinterpret_cast<DWORD*>(&data);
  int errorcode = TlsSetValue(*tls, const_cast<void*>(d));
  assert(errorcode != 0);
  (void)errorcode;
}

void ThreadLocalImpl::removeInstance() {
  setInstance(0);
}

}
