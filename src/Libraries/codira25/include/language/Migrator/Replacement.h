/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

//===--- Replacement.h - Migrator Replacements ------------------*- C++ -*-===//
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

#ifndef LANGUAGE_MIGRATOR_REPLACEMENT_H
#define LANGUAGE_MIGRATOR_REPLACEMENT_H
namespace language {
namespace migrator {

struct Replacement {
  size_t Offset;
  size_t Remove;
  std::string Text;

  bool isRemove() const {
    return Remove > 0;
  }

  bool isInsert() const { return Remove == 0 && !Text.empty(); }

  bool isReplace() const { return Remove > 0 && !Text.empty(); }

  size_t endOffset() const {
    if (isInsert()) {
      return Offset + Text.size();
    } else {
      return Offset + Remove;
    }
  }

  bool operator<(const Replacement &Other) const {
    return Offset < Other.Offset;
  }

  bool operator==(const Replacement &Other) const {
    return Offset == Other.Offset && Remove == Other.Remove &&
      Text == Other.Text;
  }
};

} // end namespace migrator
} // end namespace language

#endif // LANGUAGE_MIGRATOR_REPLACEMENT_H
