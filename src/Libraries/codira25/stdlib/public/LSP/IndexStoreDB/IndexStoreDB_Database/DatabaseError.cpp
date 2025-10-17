/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

//===--- DatabaseError.cpp ------------------------------------------------===//
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

#include <IndexStoreDB_Database/DatabaseError.h>
#include "lmdb/lmdb++.h"
#include <IndexStoreDB_LLVMSupport/toolchain_Support_raw_ostream.h>

using namespace IndexStoreDB;
using namespace IndexStoreDB::db;

void DatabaseError::raise(const char* const origin, const int rc) {
  // Move exceptions from 'lmdb++.h' to 'DatabaseError.h' as needed.
  switch (rc) {
    case MDB_KEYEXIST:         throw lmdb::key_exist_error{origin, rc};
    case MDB_NOTFOUND:         throw lmdb::not_found_error{origin, rc};
    case MDB_CORRUPTED:        throw lmdb::corrupted_error{origin, rc};
    case MDB_PANIC:            throw lmdb::panic_error{origin, rc};
    case MDB_VERSION_MISMATCH: throw lmdb::version_mismatch_error{origin, rc};
    case MDB_MAP_FULL:         throw MapFullError{origin, rc};
#ifdef MDB_BAD_DBI
    case MDB_BAD_DBI:          throw lmdb::bad_dbi_error{origin, rc};
#endif
    default:                   throw lmdb::runtime_error{origin, rc};
  }
}

const char* DatabaseError::what() const noexcept {
  return ::mdb_strerror(code());
}

std::string DatabaseError::description() const noexcept {
  std::string desc;
  toolchain::raw_string_ostream OS(desc);
  OS << origin() << ": " << what();
  return OS.str();
}

void MapFullError::_anchor() {}
