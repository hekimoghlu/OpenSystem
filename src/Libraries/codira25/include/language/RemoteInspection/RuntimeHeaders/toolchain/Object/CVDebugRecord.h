/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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

//===- CVDebugRecord.h ------------------------------------------*- C++ -*-===//
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

#ifndef TOOLCHAIN_OBJECT_CVDEBUGRECORD_H
#define TOOLCHAIN_OBJECT_CVDEBUGRECORD_H

#include "toolchain/Support/Endian.h"

namespace toolchain {
namespace OMF {
struct Signature {
  enum ID : uint32_t {
    PDB70 = 0x53445352, // RSDS
    PDB20 = 0x3031424e, // NB10
    CV50 = 0x3131424e,  // NB11
    CV41 = 0x3930424e,  // NB09
  };

  support::ulittle32_t CVSignature;
  support::ulittle32_t Offset;
};
}

namespace codeview {
struct PDB70DebugInfo {
  support::ulittle32_t CVSignature;
  uint8_t Signature[16];
  support::ulittle32_t Age;
  // char PDBFileName[];
};

struct PDB20DebugInfo {
  support::ulittle32_t CVSignature;
  support::ulittle32_t Offset;
  support::ulittle32_t Signature;
  support::ulittle32_t Age;
  // char PDBFileName[];
};

union DebugInfo {
  struct OMF::Signature Signature;
  struct PDB20DebugInfo PDB20;
  struct PDB70DebugInfo PDB70;
};
}
}

#endif

