/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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

//===- StringMatcher.h - Generate a matcher for input strings ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the StringMatcher class.
//
//===----------------------------------------------------------------------===//

#ifndef STRINGMATCHER_H
#define STRINGMATCHER_H

#include <vector>
#include <string>
#include <utility>
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class raw_ostream;
  
/// StringMatcher - Given a list of strings and code to execute when they match,
/// output a simple switch tree to classify the input string.
/// 
/// If a match is found, the code in Vals[i].second is executed; control must
/// not exit this code fragment.  If nothing matches, execution falls through.
///
class StringMatcher {
public:
  typedef std::pair<std::string, std::string> StringPair;
private:
  StringRef StrVariableName;
  const std::vector<StringPair> &Matches;
  raw_ostream &OS;
  
public:
  StringMatcher(StringRef strVariableName, 
                const std::vector<StringPair> &matches, raw_ostream &os)
    : StrVariableName(strVariableName), Matches(matches), OS(os) {}
  
  void Emit(unsigned Indent = 0) const;
  
  
private:
  bool EmitStringMatcherForChar(const std::vector<const StringPair*> &Matches,
                                unsigned CharNo, unsigned IndentCount) const;
};

} // end llvm namespace.

#endif
