/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

//===- SearchForAddressOfSpecialSymbol.cpp - Function addresses -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file pulls the addresses of certain symbols out of the linker.  It must
//  include as few header files as possible because it declares the symbols as
//  void*, which would conflict with the actual symbol type if any header
//  declared it.
//
//===----------------------------------------------------------------------===//

#include <string.h>

// Must declare the symbols in the global namespace.
static void *DoSearch(const char* symbolName) {
#define EXPLICIT_SYMBOL(SYM) \
   extern void *SYM; if (!strcmp(symbolName, #SYM)) return &SYM

  // If this is darwin, it has some funky issues, try to solve them here.  Some
  // important symbols are marked 'private external' which doesn't allow
  // SearchForAddressOfSymbol to find them.  As such, we special case them here,
  // there is only a small handful of them.

#ifdef __APPLE__
  {
    // __eprintf is sometimes used for assert() handling on x86.
    //
    // FIXME: Currently disabled when using Clang, as we don't always have our
    // runtime support libraries available.
#ifndef __clang__
#ifdef __i386__
    EXPLICIT_SYMBOL(__eprintf);
#endif
#endif
  }
#endif

#ifdef __CYGWIN__
  {
    EXPLICIT_SYMBOL(_alloca);
    EXPLICIT_SYMBOL(__main);
  }
#endif

#undef EXPLICIT_SYMBOL
  return 0;
}

namespace llvm {
void *SearchForAddressOfSpecialSymbol(const char* symbolName) {
  return DoSearch(symbolName);
}
}  // namespace llvm
