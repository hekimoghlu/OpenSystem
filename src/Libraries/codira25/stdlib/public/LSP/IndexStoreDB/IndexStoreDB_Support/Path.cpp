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

//===--- Path.cpp ---------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#include <IndexStoreDB_Support/Path.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringMap.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Allocator.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Path.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_FileSystem.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Mutex.h>
#if defined(_WIN32)
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <limits.h>
#include <stdlib.h>

#if defined(_WIN32)
#define PATH_MAX MAX_PATH
#endif

using namespace IndexStoreDB;

namespace {
class CanonicalPathCacheImpl {
  toolchain::StringMap<CanonicalFilePathRef, toolchain::BumpPtrAllocator> CanonPaths;
  mutable toolchain::sys::Mutex StateMtx;

public:
  CanonicalFilePath getCanonicalPath(StringRef Path,
                                     StringRef WorkingDir = StringRef());
};
}

CanonicalFilePath
CanonicalPathCacheImpl::getCanonicalPath(StringRef Path, StringRef WorkingDir) {
  if (Path.empty())
    return CanonicalFilePath();

  SmallString<256> AbsPath;
  if (toolchain::sys::path::is_absolute(Path)) {
    AbsPath = Path;
  } else {
    assert(!WorkingDir.empty() && "passed relative path without working-dir");
    AbsPath = WorkingDir;
    AbsPath += '/';
    AbsPath += Path;
  }

  {
    toolchain::sys::ScopedLock L(StateMtx);
    auto It = CanonPaths.find(AbsPath);
    if (It != CanonPaths.end())
      return It->second;
  }

  toolchain::SmallString<PATH_MAX> Buffer;
  if (toolchain::sys::fs::real_path(AbsPath.c_str(), Buffer, false)) {
    return CanonicalFilePathRef::getAsCanonicalPath(AbsPath);
  }
  StringRef CanonPath = Buffer;

  {
    toolchain::sys::ScopedLock L(StateMtx);
    auto Pair = CanonPaths.insert(std::make_pair(AbsPath.str(), CanonicalFilePathRef()));
    auto &It = Pair.first;
    bool WasInserted = Pair.second;
    if (!WasInserted)
      return It->second;

    CanonicalFilePathRef CanonPathRef;
    if (CanonPath == It->first()) {
      CanonPathRef = CanonicalFilePathRef::getAsCanonicalPath(It->first());
    } else {
      auto &Alloc = CanonPaths.getAllocator();
      char *CopyPtr = Alloc.Allocate<char>(CanonPath.size());
      std::uninitialized_copy(CanonPath.begin(), CanonPath.end(), CopyPtr);
      StringRef CopyCanonPath(CopyPtr, CanonPath.size());
      CanonPathRef = CanonicalFilePathRef::getAsCanonicalPath(CopyCanonPath);
    }
    It->second = CanonPathRef;
    return CanonPathRef;
  }
}


CanonicalPathCache::CanonicalPathCache() {
  Impl = new CanonicalPathCacheImpl();
}
CanonicalPathCache::~CanonicalPathCache() {
  delete static_cast<CanonicalPathCacheImpl*>(Impl);
}

CanonicalFilePath
CanonicalPathCache::getCanonicalPath(StringRef Path, StringRef WorkingDir) {
  return static_cast<CanonicalPathCacheImpl*>(Impl)->getCanonicalPath(Path, WorkingDir);
}
