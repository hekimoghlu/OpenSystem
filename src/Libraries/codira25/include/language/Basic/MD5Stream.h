/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

//===--- MD5Stream.h - raw_ostream that compute MD5  ------------*- C++ -*-===//
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

#ifndef LANGUAGE_MD5STREAM_H
#define LANGUAGE_MD5STREAM_H

#include "toolchain/Support/MD5.h"
#include "toolchain/Support/raw_ostream.h"

namespace language {

/// An output stream which calculates the MD5 hash of the streamed data.
class MD5Stream : public toolchain::raw_ostream {
private:

  uint64_t Pos = 0;
  toolchain::MD5 Hash;

  void write_impl(const char *Ptr, size_t Size) override {
    Hash.update(ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Ptr), Size));
    Pos += Size;
  }

  uint64_t current_pos() const override { return Pos; }

public:

  void final(toolchain::MD5::MD5Result &Result) {
    flush();
    Hash.final(Result);
  }
};

} // namespace language

#endif
