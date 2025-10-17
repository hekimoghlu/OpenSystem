/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#include "AirSpecial.h"

#if ENABLE(B3_JIT)

#include <limits.h>
#include <wtf/StringPrintStream.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 { namespace Air {

const char* const Special::dumpPrefix = "&";

WTF_MAKE_TZONE_ALLOCATED_IMPL(Special);

Special::Special() = default;

Special::~Special() = default;

CString Special::name() const
{
    StringPrintStream out;
    dumpImpl(out);
    return out.toCString();
}

std::optional<unsigned> Special::shouldTryAliasingDef(Inst&)
{
    return std::nullopt;
}

bool Special::isTerminal(Inst&)
{
    return false;
}

bool Special::hasNonArgEffects(Inst&)
{
    return true;
}

bool Special::hasNonArgNonControlEffects(Inst&)
{
    return true;
}

void Special::dump(PrintStream& out) const
{
    out.print(dumpPrefix);
    dumpImpl(out);
    if (m_index != UINT_MAX)
        out.print(m_index);
}

void Special::deepDump(PrintStream& out) const
{
    out.print(*this, ": ");
    deepDumpImpl(out);
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
