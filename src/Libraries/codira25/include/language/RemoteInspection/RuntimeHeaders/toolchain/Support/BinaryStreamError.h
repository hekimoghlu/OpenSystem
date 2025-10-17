/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

//===- BinaryStreamError.h - Error extensions for Binary Streams *- C++ -*-===//
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

#ifndef TOOLCHAIN_SUPPORT_BINARYSTREAMERROR_H
#define TOOLCHAIN_SUPPORT_BINARYSTREAMERROR_H

#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/Error.h"

#include <string>

namespace toolchain {
enum class stream_error_code {
  unspecified,
  stream_too_short,
  invalid_array_size,
  invalid_offset,
  filesystem_error
};

/// Base class for errors originating when parsing raw PDB files
class BinaryStreamError : public ErrorInfo<BinaryStreamError> {
public:
  static char ID;
  explicit BinaryStreamError(stream_error_code C);
  explicit BinaryStreamError(StringRef Context);
  BinaryStreamError(stream_error_code C, StringRef Context);

  void log(raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  StringRef getErrorMessage() const;

  stream_error_code getErrorCode() const { return Code; }

private:
  std::string ErrMsg;
  stream_error_code Code;
};
} // namespace toolchain

#endif // TOOLCHAIN_SUPPORT_BINARYSTREAMERROR_H
