/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

//===--- StoreSymbolRecord.h ------------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_LIB_INDEX_STORESYMBOLRECORD_H
#define INDEXSTOREDB_LIB_INDEX_STORESYMBOLRECORD_H

#include <IndexStoreDB_Core/Symbol.h>
#include <IndexStoreDB_Index/SymbolDataProvider.h>
#include <IndexStoreDB_Database/IDCode.h>
#include <IndexStoreDB_Index/IndexStoreCXX.h>
#include <IndexStoreDB_LLVMSupport/toolchain_ADT_StringRef.h>
#include <string>

namespace IndexStoreDB {
  class CanonicalFilePathRef;
  class TimestampedPath;

namespace index {
  class StoreSymbolRecord;
  typedef std::shared_ptr<StoreSymbolRecord> StoreSymbolRecordRef;

struct FileAndTarget {
  TimestampedPath Path;
  std::string Target;
};

class StoreSymbolRecord : public SymbolDataProvider {
  indexstore::IndexStoreRef Store;
  std::string RecordName;
  db::IDCode ProviderCode;
  SymbolProviderKind SymProviderKind;
  std::vector<FileAndTarget> FileAndTargetRefs;

public:
  ~StoreSymbolRecord();

  static StoreSymbolRecordRef create(indexstore::IndexStoreRef store,
                                     StringRef recordName, db::IDCode providerCode,
                                     SymbolProviderKind symProviderKind,
                                     ArrayRef<FileAndTarget> fileReferences);

  StringRef getName() const { return RecordName; }

  SymbolProviderKind getProviderKind() const {
    return SymProviderKind;
  }

  ArrayRef<FileAndTarget> getSourceFileReferencesAndTargets() const {
    return FileAndTargetRefs;
  }

  /// \returns true for error.
  bool doForData(function_ref<void(indexstore::IndexRecordReader &)> Action);

  //===--------------------------------------------------------------------===//
  // SymbolDataProvider interface
  //===--------------------------------------------------------------------===//

  virtual std::string getIdentifier() const override { return RecordName; }

  virtual bool isSystem() const override;

  virtual bool foreachCoreSymbolData(function_ref<bool(StringRef USR,
                                                       StringRef Name,
                                                       SymbolInfo Info,
                                                       SymbolRoleSet Roles,
                                                       SymbolRoleSet RelatedRoles)> Receiver) override;

  virtual bool foreachSymbolOccurrence(function_ref<bool(SymbolOccurrenceRef Occur)> Receiver) override;

  virtual bool foreachSymbolOccurrenceByUSR(ArrayRef<db::IDCode> USRs,
                                            SymbolRoleSet RoleSet,
               function_ref<bool(SymbolOccurrenceRef Occur)> Receiver) override;

  virtual bool foreachRelatedSymbolOccurrenceByUSR(ArrayRef<db::IDCode> USRs,
                                            SymbolRoleSet RoleSet,
               function_ref<bool(SymbolOccurrenceRef Occur)> Receiver) override;

  virtual bool foreachUnitTestSymbolOccurrence(
               function_ref<bool(SymbolOccurrenceRef Occur)> Receiver) override;

};

} // namespace index
} // namespace IndexStoreDB

#endif
