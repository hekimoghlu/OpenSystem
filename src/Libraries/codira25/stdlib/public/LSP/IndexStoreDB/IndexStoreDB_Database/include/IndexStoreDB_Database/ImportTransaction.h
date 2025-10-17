/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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

//===--- ImportTransaction.h ------------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_SKDATABASE_IMPORTTRANSACTION_H
#define INDEXSTOREDB_SKDATABASE_IMPORTTRANSACTION_H

#include <IndexStoreDB_Core/Symbol.h>
#include <IndexStoreDB_Database/UnitInfo.h>
#include <IndexStoreDB_Support/Path.h>
#include <memory>
#include <unordered_set>

namespace IndexStoreDB {
namespace db {
  class Database;
  typedef std::shared_ptr<Database> DatabaseRef;

class INDEXSTOREDB_EXPORT ImportTransaction {
public:
  explicit ImportTransaction(DatabaseRef dbase);
  ~ImportTransaction();

  IDCode getUnitCode(StringRef unitName);
  IDCode addProviderName(StringRef name, bool *wasInserted = nullptr);
  // Marks a provider as containing test symbols.
  void setProviderContainsTestSymbols(IDCode provider);
  bool providerContainsTestSymbols(IDCode provider);
  /// \returns a IDCode of the USR.
  IDCode addSymbolInfo(IDCode provider,
                       StringRef USR, StringRef symbolName, SymbolInfo symInfo,
                       SymbolRoleSet roles, SymbolRoleSet relatedRoles);
  IDCode addFilePath(CanonicalFilePathRef filePath);
  IDCode addUnitFileIdentifier(StringRef unitFile);

  void removeUnitData(IDCode unitCode);
  void removeUnitData(StringRef unitName);

  void commit();

  class Implementation;
  Implementation *_impl() const { return Impl.get(); }
private:
  std::unique_ptr<Implementation> Impl;
};

class INDEXSTOREDB_EXPORT UnitDataImport {
  ImportTransaction &Import;
  std::string UnitName;
  CanonicalFilePath MainFile;
  std::string OutFileIdentifier;
  CanonicalFilePath Sysroot;
  toolchain::sys::TimePoint<> ModTime;
  Optional<bool> IsSystem;
  Optional<bool> HasTestSymbols;
  Optional<SymbolProviderKind> SymProviderKind;
  std::string Target;

  IDCode UnitCode;
  bool IsMissing;
  bool IsUpToDate;
  IDCode PrevMainFileCode;
  IDCode PrevOutFileCode;
  IDCode PrevSysrootCode;
  IDCode PrevTargetCode;
  std::unordered_set<IDCode> PrevCombinedFileDepends; // Combines record and non-record file dependencies.
  std::unordered_set<IDCode> PrevUnitDepends;
  std::unordered_set<UnitInfo::Provider> PrevProviderDepends;

  std::vector<IDCode> FileDepends;
  std::vector<IDCode> UnitDepends;
  std::vector<UnitInfo::Provider> ProviderDepends;

public:
  UnitDataImport(ImportTransaction &import, StringRef unitName, toolchain::sys::TimePoint<> modTime);
  ~UnitDataImport();

  IDCode getUnitCode() const { return UnitCode; }

  bool isMissing() const { return IsMissing; }
  bool isUpToDate() const { return IsUpToDate; }
  Optional<bool> getIsSystem() const { return IsSystem; }
  Optional<bool> getHasTestSymbols() const { return HasTestSymbols; }
  Optional<SymbolProviderKind> getSymbolProviderKind() const { return SymProviderKind; }

  IDCode getPrevMainFileCode() const {
    assert(!IsMissing);
    return PrevMainFileCode;
  }
  IDCode getPrevOutFileCode() const {
    assert(!IsMissing);
    return PrevOutFileCode;
  }

  void setMainFile(CanonicalFilePathRef mainFile);
  void setOutFileIdentifier(StringRef outFileIdentifier);
  void setSysroot(CanonicalFilePathRef sysroot);
  void setIsSystemUnit(bool isSystem);
  void setSymbolProviderKind(SymbolProviderKind K);
  void setTarget(StringRef target);

  IDCode addFileDependency(CanonicalFilePathRef filePathDep);
  IDCode addUnitDependency(StringRef unitNameDep);
  /// \returns the provider code.
  IDCode addProviderDependency(StringRef providerName, CanonicalFilePathRef filePathDep, StringRef moduleName, bool isSystem, bool *isNewProvider = nullptr);

  void commit();
};

} // namespace db
} // namespace IndexStoreDB

#endif
