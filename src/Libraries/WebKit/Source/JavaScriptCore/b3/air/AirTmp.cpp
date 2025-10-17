/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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
#include "AirTmp.h"

#if ENABLE(B3_JIT)

#include "AirTmpInlines.h"

namespace JSC { namespace B3 { namespace Air {

template<> const char* const Tmp::Indexed<GP>::dumpPrefix = "%tmp";
template<> const char* const Tmp::Indexed<FP>::dumpPrefix = "%ftmp";
template<> const char* const Tmp::AbsolutelyIndexed<GP>::dumpPrefix = "%abs";
template<> const char* const Tmp::AbsolutelyIndexed<FP>::dumpPrefix = "%fabs";
const char* const Tmp::LinearlyIndexed::dumpPrefix = "%ltmp";

void Tmp::dump(PrintStream& out) const
{
    if (!*this) {
        out.print("<none>");
        return;
    }

    if (isReg()) {
        out.print(reg());
        return;
    }

    if (isGP()) {
        out.print("%tmp", gpTmpIndex());
        return;
    }

    out.print("%ftmp", fpTmpIndex());
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
