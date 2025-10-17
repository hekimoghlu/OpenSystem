/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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

//===--- Options.cpp - Option info & table --------------------------------===//
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

#include "language/Option/Options.h"

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/Option/OptTable.h"
#include "toolchain/Option/Option.h"

using namespace language::options;
using namespace toolchain::opt;

#define PREFIX(NAME, VALUE)                                                    \
  constexpr toolchain::StringLiteral NAME##_init[] = VALUE;                         \
  constexpr toolchain::ArrayRef<toolchain::StringLiteral> NAME(                          \
      NAME##_init, std::size(NAME##_init) - 1);
#include "language/Option/Options.inc"
#undef PREFIX

static const toolchain::opt::GenericOptTable::Info InfoTable[] = {
#define OPTION(...) TOOLCHAIN_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "language/Option/Options.inc"
#undef OPTION
};

namespace {

class CodiraOptTable : public toolchain::opt::GenericOptTable {
public:
  CodiraOptTable() : GenericOptTable(InfoTable) {}
};

} // end anonymous namespace

std::unique_ptr<OptTable> language::createCodiraOptTable() {
  return std::unique_ptr<GenericOptTable>(new CodiraOptTable());
}
