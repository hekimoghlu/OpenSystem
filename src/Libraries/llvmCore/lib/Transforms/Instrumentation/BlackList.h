/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 8, 2024.
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

//===-- BlackList.h - blacklist for sanitizers ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//
//
// This is a utility class for instrumentation passes (like AddressSanitizer
// or ThreadSanitizer) to avoid instrumenting some functions or global
// variables based on a user-supplied blacklist.
//
// The blacklist disables instrumentation of various functions and global
// variables.  Each line contains a prefix, followed by a wild card expression.
// ---
// fun:*_ZN4base6subtle*
// global:*global_with_bad_access_or_initialization*
// global-init:*global_with_initialization_issues*
// src:file_with_tricky_code.cc
// ---
// Note that the wild card is in fact an llvm::Regex, but * is automatically
// replaced with .*
// This is similar to the "ignore" feature of ThreadSanitizer.
// http://code.google.com/p/data-race-test/wiki/ThreadSanitizerIgnores
//
//===----------------------------------------------------------------------===//
//

#include "llvm/ADT/StringMap.h"

namespace llvm {
class Function;
class GlobalVariable;
class Module;
class Regex;
class StringRef;

class BlackList {
 public:
  BlackList(const StringRef Path);
  // Returns whether either this function or it's source file are blacklisted.
  bool isIn(const Function &F);
  // Returns whether either this global or it's source file are blacklisted.
  bool isIn(const GlobalVariable &G);
  // Returns whether this module is blacklisted by filename.
  bool isIn(const Module &M);
  // Returns whether a global should be excluded from initialization checking.
  bool isInInit(const GlobalVariable &G);
 private:
  StringMap<Regex*> Entries;

  bool inSection(const StringRef Section, const StringRef Query);
};

}  // namespace llvm
