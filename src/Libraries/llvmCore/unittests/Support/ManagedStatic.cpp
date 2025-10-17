/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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

//===- llvm/unittest/Support/ManagedStatic.cpp - ManagedStatic tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Config/config.h"
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#include "gtest/gtest.h"

using namespace llvm;

namespace {

#ifdef HAVE_PTHREAD_H
namespace test1 {
  llvm::ManagedStatic<int> ms;
  void *helper(void*) {
    *ms;
    return NULL;
  }
}

TEST(Initialize, MultipleThreads) {
  // Run this test under tsan: http://code.google.com/p/data-race-test/

  llvm_start_multithreaded();
  pthread_t t1, t2;
  pthread_create(&t1, NULL, test1::helper, NULL);
  pthread_create(&t2, NULL, test1::helper, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  llvm_stop_multithreaded();
}
#endif

} // anonymous namespace
