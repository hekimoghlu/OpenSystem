/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

//===--- StoreUnitInfo.h ----------------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_INDEX_STOREUNITINFO_H
#define INDEXSTOREDB_INDEX_STOREUNITINFO_H

#include <IndexStoreDB_Support/Path.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Chrono.h>
#include <string>

namespace IndexStoreDB {
namespace index {

struct StoreUnitInfo {
  std::string UnitName;
  CanonicalFilePath MainFilePath;
  std::string OutFileIdentifier;
  bool HasTestSymbols = false;
  toolchain::sys::TimePoint<> ModTime;

  StoreUnitInfo() = default;
  StoreUnitInfo(std::string unitName, CanonicalFilePath mainFilePath,
                StringRef outFileIdentifier, bool hasTestSymbols,
                toolchain::sys::TimePoint<> modTime)
      : UnitName(unitName),
        MainFilePath(mainFilePath),
        OutFileIdentifier(outFileIdentifier),
        HasTestSymbols(hasTestSymbols),
        ModTime(modTime) {}
};

} // namespace index
} // namespace IndexStoreDB

#endif
