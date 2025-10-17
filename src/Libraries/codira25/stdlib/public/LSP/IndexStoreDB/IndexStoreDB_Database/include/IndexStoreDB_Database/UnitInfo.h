/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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

//===--- UnitInfo.h ---------------------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_SKDATABASE_UNITINFO_H
#define INDEXSTOREDB_SKDATABASE_UNITINFO_H

#include <IndexStoreDB_Database/IDCode.h>
#include <IndexStoreDB_Support/LLVM.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_ArrayRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_Hashing.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>
#include <IndexStoreDB_LLVMSupport/toolchain_Support_Chrono.h>

namespace IndexStoreDB {
namespace db {

struct UnitInfo {
  struct Provider {
    IDCode ProviderCode;
    IDCode FileCode;

    friend bool operator ==(const Provider &lhs, const Provider &rhs) {
      return lhs.ProviderCode == rhs.ProviderCode && lhs.FileCode == rhs.FileCode;
    }
    friend bool operator !=(const Provider &lhs, const Provider &rhs) {
      return !(lhs == rhs);
    }
  };

  StringRef UnitName;
  IDCode UnitCode;
  toolchain::sys::TimePoint<> ModTime;
  IDCode OutFileCode;
  IDCode MainFileCode;
  IDCode SysrootCode;
  IDCode TargetCode;
  bool HasMainFile;
  bool HasSysroot;
  bool IsSystem;
  bool HasTestSymbols;
  SymbolProviderKind SymProviderKind;
  ArrayRef<IDCode> FileDepends;
  ArrayRef<IDCode> UnitDepends;
  ArrayRef<Provider> ProviderDepends;

  bool isInvalid() const { return UnitName.empty(); }
  bool isValid() const { return !isInvalid(); }
};

} // namespace db
} // namespace IndexStoreDB

namespace std {
template <> struct hash<IndexStoreDB::db::UnitInfo::Provider> {
  size_t operator()(const IndexStoreDB::db::UnitInfo::Provider &k) const {
    return toolchain::hash_combine(k.FileCode.value(), k.ProviderCode.value());
  }
};
}

#endif
