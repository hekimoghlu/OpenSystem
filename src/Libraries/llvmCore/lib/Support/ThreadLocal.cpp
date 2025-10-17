/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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

//===- ThreadLocal.cpp - Thread Local Data ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the llvm::sys::ThreadLocal class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/ThreadLocal.h"

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

#if !defined(LLVM_ENABLE_THREADS) || LLVM_ENABLE_THREADS == 0
// Define all methods as no-ops if threading is explicitly disabled
namespace llvm {
using namespace sys;
ThreadLocalImpl::ThreadLocalImpl() { }
ThreadLocalImpl::~ThreadLocalImpl() { }
void ThreadLocalImpl::setInstance(const void* d) {
  typedef int SIZE_TOO_BIG[sizeof(d) <= sizeof(data) ? 1 : -1];
  void **pd = reinterpret_cast<void**>(&data);
  *pd = const_cast<void*>(d);
}
const void* ThreadLocalImpl::getInstance() {
  void **pd = reinterpret_cast<void**>(&data);
  return *pd;
}
void ThreadLocalImpl::removeInstance() {
  setInstance(0);
}
}
#else

#if defined(HAVE_PTHREAD_H) && defined(HAVE_PTHREAD_GETSPECIFIC)

#include <cassert>
#include <pthread.h>
#include <stdlib.h>

namespace llvm {
using namespace sys;

ThreadLocalImpl::ThreadLocalImpl() : data() {
  typedef int SIZE_TOO_BIG[sizeof(pthread_key_t) <= sizeof(data) ? 1 : -1];
  pthread_key_t* key = reinterpret_cast<pthread_key_t*>(&data);
  int errorcode = pthread_key_create(key, NULL);
  assert(errorcode == 0);
  (void) errorcode;
}

ThreadLocalImpl::~ThreadLocalImpl() {
  pthread_key_t* key = reinterpret_cast<pthread_key_t*>(&data);
  int errorcode = pthread_key_delete(*key);
  assert(errorcode == 0);
  (void) errorcode;
}

void ThreadLocalImpl::setInstance(const void* d) {
  pthread_key_t* key = reinterpret_cast<pthread_key_t*>(&data);
  int errorcode = pthread_setspecific(*key, d);
  assert(errorcode == 0);
  (void) errorcode;
}

const void* ThreadLocalImpl::getInstance() {
  pthread_key_t* key = reinterpret_cast<pthread_key_t*>(&data);
  return pthread_getspecific(*key);
}

void ThreadLocalImpl::removeInstance() {
  setInstance(0);
}

}

#elif defined(LLVM_ON_UNIX)
#include "Unix/ThreadLocal.inc"
#elif defined( LLVM_ON_WIN32)
#include "Windows/ThreadLocal.inc"
#else
#warning Neither LLVM_ON_UNIX nor LLVM_ON_WIN32 set in Support/ThreadLocal.cpp
#endif
#endif
