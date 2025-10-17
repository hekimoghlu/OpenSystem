/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

//===-- JIT.h - Abstract Execution Engine Interface -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Common functionality for JITEventListener implementations
//
//===----------------------------------------------------------------------===//

#ifndef EVENT_LISTENER_COMMON_H
#define EVENT_LISTENER_COMMON_H

#include "llvm/DebugInfo.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/Path.h"

namespace llvm {

namespace jitprofiling {

class FilenameCache {
  // Holds the filename of each Scope, so that we can pass a null-terminated
  // string into oprofile.  Use an AssertingVH rather than a ValueMap because we
  // shouldn't be modifying any MDNodes while this map is alive.
  DenseMap<AssertingVH<MDNode>, std::string> Filenames;
  DenseMap<AssertingVH<MDNode>, std::string> Paths;

 public:
  const char *getFilename(MDNode *Scope) {
    std::string &Filename = Filenames[Scope];
    if (Filename.empty()) {
      DIScope DIScope(Scope);
      Filename = DIScope.getFilename();
    }
    return Filename.c_str();
  }

  const char *getFullPath(MDNode *Scope) {
    std::string &P = Paths[Scope];
    if (P.empty()) {
      DIScope DIScope(Scope);
      StringRef DirName = DIScope.getDirectory();
      StringRef FileName = DIScope.getFilename();
      SmallString<256> FullPath;
      if (DirName != "." && DirName != "") {
        FullPath = DirName;
      }
      if (FileName != "") {
        sys::path::append(FullPath, FileName);
      }
      P = FullPath.str();
    }
    return P.c_str();
  }
};

} // namespace jitprofiling

} // namespace llvm

#endif //EVENT_LISTENER_COMMON_H
