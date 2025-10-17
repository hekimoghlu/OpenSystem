/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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

//===--- DatabaseError.h ----------------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_SKDATABASE_DATABASEERROR_H
#define INDEXSTOREDB_SKDATABASE_DATABASEERROR_H

#include <IndexStoreDB_Support/Visibility.h>
#include <string>
#include <stdexcept>

namespace IndexStoreDB {
namespace db {

class INDEXSTOREDB_EXPORT DatabaseError : public std::runtime_error {
protected:
  const int _code;

public:
  /// Throws an error based on the given return code.
  [[noreturn]] static void raise(const char* origin, int rc);

  DatabaseError(const char* const origin, const int rc) noexcept
    : runtime_error{origin}, _code{rc} {}

  /// Returns the underlying error code.
  int code() const noexcept {
    return _code;
  }

  /// Returns the origin of the error.
  const char* origin() const noexcept {
    return runtime_error::what();
  }

  /// Returns the underlying error message.
  virtual const char* what() const noexcept override;

  std::string description() const noexcept;
};

/// Exception class for `MDB_MAP_FULL` errors.
///
/// @see http://symas.com/mdb/doc/group__errors.html#ga0a83370402a060c9175100d4bbfb9f25
///
class INDEXSTOREDB_EXPORT MapFullError final : public DatabaseError {
  virtual void _anchor();
public:
  using DatabaseError::DatabaseError;
};

} // namespace db
} // namespace IndexStoreDB

#endif
