/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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

//===--- Fingerprint.cpp - A stable identity for compiler data --*- C++ -*-===//
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

#include "language/Basic/Assertions.h"
#include "language/Basic/Fingerprint.h"
#include "language/Basic/STLExtras.h"
#include "toolchain/Support/Format.h"
#include "toolchain/Support/raw_ostream.h"

#include <inttypes.h>
#include <sstream>

using namespace language;

toolchain::raw_ostream &toolchain::operator<<(toolchain::raw_ostream &OS,
                                    const Fingerprint &FP) {
  return OS << FP.getRawValue();
}

void language::simple_display(toolchain::raw_ostream &out, const Fingerprint &fp) {
  out << fp.getRawValue();
}

std::optional<Fingerprint> Fingerprint::fromString(toolchain::StringRef value) {
  assert(value.size() == Fingerprint::DIGEST_LENGTH &&
         "Only supports 32-byte hash values!");
  auto fp = Fingerprint::ZERO();
  {
    std::istringstream s(value.drop_back(Fingerprint::DIGEST_LENGTH/2).str());
    s >> std::hex >> fp.core.first;
  }
  {
    std::istringstream s(value.drop_front(Fingerprint::DIGEST_LENGTH/2).str());
    s >> std::hex >> fp.core.second;
  }
  // If the input string is not valid hex, the conversion above can fail.
  if (value != fp.getRawValue())
    return std::nullopt;

  return fp;
}

toolchain::SmallString<Fingerprint::DIGEST_LENGTH> Fingerprint::getRawValue() const {
  toolchain::SmallString<Fingerprint::DIGEST_LENGTH> Str;
  toolchain::raw_svector_ostream Res(Str);
  Res << toolchain::format_hex_no_prefix(core.first, 16);
  Res << toolchain::format_hex_no_prefix(core.second, 16);
  return Str;
}
