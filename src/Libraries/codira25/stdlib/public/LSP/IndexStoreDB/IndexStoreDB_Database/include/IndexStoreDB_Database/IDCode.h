/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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

//===--- IDCode.h -----------------------------------------------*- C++ -*-===//
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

#ifndef INDEXSTOREDB_SKDATABASE_IDCODE_H
#define INDEXSTOREDB_SKDATABASE_IDCODE_H

#include <functional>
#include <cstdint>

namespace IndexStoreDB {
namespace db {

class IDCode {
  uint64_t Code{};
  explicit IDCode(uint64_t code) : Code(code) {}

public:
  IDCode() = default;

  static IDCode fromValue(uint64_t code) {
    return IDCode(code);
  }

  uint64_t value() const { return Code; }

  friend bool operator ==(IDCode lhs, IDCode rhs) {
    return lhs.Code == rhs.Code;
  }
  friend bool operator !=(IDCode lhs, IDCode rhs) {
    return !(lhs == rhs);
  }

  static int compare(IDCode lhs, IDCode rhs) {
    if (lhs.value() < rhs.value()) return -1;
    if (lhs.value() > rhs.value()) return 1;
    return 0;
  }
};

} // namespace db
} // namespace IndexStoreDB

namespace std {
template <> struct hash<IndexStoreDB::db::IDCode> {
  size_t operator()(const IndexStoreDB::db::IDCode &k) const {
    return k.value();
  }
};
}

#endif
