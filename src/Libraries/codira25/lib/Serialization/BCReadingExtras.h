/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

//===--- BCReadingExtras.h - Useful helpers for bitcode reading -*- C++ -*-===//
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

#ifndef LANGUAGE_SERIALIZATION_BCREADINGEXTRAS_H
#define LANGUAGE_SERIALIZATION_BCREADINGEXTRAS_H

#include "toolchain/Bitstream/BitstreamReader.h"

namespace language {
namespace serialization {

/// Saves and restores a BitstreamCursor's bit offset in its stream.
class BCOffsetRAII {
  toolchain::BitstreamCursor *Cursor;
  decltype(Cursor->GetCurrentBitNo()) Offset;

public:
  explicit BCOffsetRAII(toolchain::BitstreamCursor &cursor)
    : Cursor(&cursor), Offset(cursor.GetCurrentBitNo()) {}

  void reset() {
    if (Cursor)
      Offset = Cursor->GetCurrentBitNo();
  }

  void cancel() {
    Cursor = nullptr;
  }

  ~BCOffsetRAII() {
    if (Cursor)
      cantFail(Cursor->JumpToBit(Offset),
               "BCOffsetRAII must be able to go back");
  }
};

} // end namespace serialization
} // end namespace language

static constexpr const auto AF_DontPopBlockAtEnd =
  toolchain::BitstreamCursor::AF_DontPopBlockAtEnd;

#endif
