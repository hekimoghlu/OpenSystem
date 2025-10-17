/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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

//===--- FileVisibilityChecker.h --------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_LIB_INDEX_FILEVISIBILITYCHECKER_H
#define INDEXSTOREDB_LIB_INDEX_FILEVISIBILITYCHECKER_H

#include <IndexStoreDB_Database/IDCode.h>
#include <IndexStoreDB_Support/LLVM.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Mutex.h>
#include <unordered_map>
#include <unordered_set>

namespace IndexStoreDB {
  class CanonicalPathCache;

namespace db {
  class IDCode;
  class ReadTransaction;
  class Database;
  typedef std::shared_ptr<Database> DatabaseRef;
  struct UnitInfo;
}

namespace index {

class FileVisibilityChecker {
  db::DatabaseRef DBase;
  std::shared_ptr<CanonicalPathCache> CanonPathCache;

  mutable toolchain::sys::Mutex VisibleCacheMtx;
  std::unordered_set<db::IDCode> VisibleMainFiles;
  std::unordered_map<db::IDCode, unsigned> MainFilesRefCount;
  std::unordered_map<db::IDCode, bool> UnitVisibilityCache;

  std::unordered_set<db::IDCode> OutUnitFiles;
  bool UseExplicitOutputUnits;

public:
  FileVisibilityChecker(db::DatabaseRef dbase,
                        std::shared_ptr<CanonicalPathCache> canonPathCache,
                        bool useExplicitOutputUnits);

  void registerMainFiles(ArrayRef<StringRef> filePaths, StringRef productName);
  void unregisterMainFiles(ArrayRef<StringRef> filePaths, StringRef productName);

  void addUnitOutFilePaths(ArrayRef<StringRef> filePaths);
  void removeUnitOutFilePaths(ArrayRef<StringRef> filePaths);

  bool isUnitVisible(const db::UnitInfo &unitInfo, db::ReadTransaction &reader);
};

} // namespace index
} // namespace IndexStoreDB

#endif
