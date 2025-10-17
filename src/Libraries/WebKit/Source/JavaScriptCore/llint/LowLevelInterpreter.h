/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#pragma once

#include "Opcode.h"

#if ENABLE(C_LOOP)

namespace JSC {

// The following is a set of alias for the opcode names. This is needed
// because there is code (e.g. in GetByStatus.cpp and PutByStatus.cpp)
// which refers to the opcodes expecting them to be prefixed with "llint_".
// In the CLoop implementation, the 2 are equivalent. Hence, we set up this
// alias here.

#define LLINT_OPCODE_ALIAS(opcode, length) \
    const OpcodeID llint_##opcode = opcode;
FOR_EACH_CORE_OPCODE_ID(LLINT_OPCODE_ALIAS)
#undef LLINT_OPCODE_ALIAS

} // namespace JSC

#endif // ENABLE(C_LOOP)
