/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#include "config.h"
#include "B3UpsilonValue.h"

#if ENABLE(B3_JIT)

namespace JSC { namespace B3 {

UpsilonValue::~UpsilonValue() = default;

void UpsilonValue::dumpMeta(CommaPrinter& comma, PrintStream& out) const
{
    if (m_phi)
        out.print(comma, "^", m_phi->index());
    else {
        // We want to have a dump for when the Phi isn't set yet, since although such IR won't pass
        // validation, we may have such IR as an intermediate step.
        out.print(comma, "^(null)");
    }
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
