/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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

#include <dlfcn.h>
#include <stdio.h>

// Mark foo and bar weak so that Clang allows the run-time linker to decide which DSO's symbol to
// use.

__attribute__((weak)) extern "C" void foo() {
  printf("foo lib1\n");
  void (*next)(void) = reinterpret_cast<void (*)()>(dlsym(RTLD_NEXT, "foo"));
  if (next != nullptr) next();
}

__attribute__((weak)) extern "C" void bar();

void lib1_call_funcs() {
  printf("lib1_call_funcs\n");
  foo();
  bar();
}
