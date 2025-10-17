/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

//===- MemoryBufferRef.h - Memory Buffer Reference --------------*- C++ -*-===//
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
//
//  This file defines the MemoryBuffer interface.
//
//===----------------------------------------------------------------------===//

#ifndef TOOLCHAIN_SUPPORT_MEMORYBUFFERREF_H
#define TOOLCHAIN_SUPPORT_MEMORYBUFFERREF_H

#include "toolchain/ADT/StringRef.h"

namespace toolchain {

class MemoryBuffer;

class MemoryBufferRef {
  StringRef Buffer;
  StringRef Identifier;

public:
  MemoryBufferRef() = default;
  MemoryBufferRef(const MemoryBuffer &Buffer);
  MemoryBufferRef(StringRef Buffer, StringRef Identifier)
      : Buffer(Buffer), Identifier(Identifier) {}

  StringRef getBuffer() const { return Buffer; }
  StringRef getBufferIdentifier() const { return Identifier; }

  const char *getBufferStart() const { return Buffer.begin(); }
  const char *getBufferEnd() const { return Buffer.end(); }
  size_t getBufferSize() const { return Buffer.size(); }

  /// Check pointer identity (not value) of identifier and data.
  friend bool operator==(const MemoryBufferRef &LHS,
                         const MemoryBufferRef &RHS) {
    return LHS.Buffer.begin() == RHS.Buffer.begin() &&
           LHS.Buffer.end() == RHS.Buffer.end() &&
           LHS.Identifier.begin() == RHS.Identifier.begin() &&
           LHS.Identifier.end() == RHS.Identifier.end();
  }

  friend bool operator!=(const MemoryBufferRef &LHS,
                         const MemoryBufferRef &RHS) {
    return !(LHS == RHS);
  }
};

} // namespace toolchain

#endif // TOOLCHAIN_SUPPORT_MEMORYBUFFERREF_H
