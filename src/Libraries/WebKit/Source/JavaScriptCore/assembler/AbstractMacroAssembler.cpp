/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#include "MacroAssembler.h" // Have to break with style because AbstractMacroAssembler.h is a shady header.

#if ENABLE(ASSEMBLER)

#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/PrintStream.h>

namespace JSC {

void AbstractMacroAssemblerBase::initializeRandom()
{
    // No strong cryptographic characteristics are necessary.
    static std::once_flag onceKey;
    static uint32_t globalCounter;
    std::call_once(onceKey, [&] {
        globalCounter = cryptographicallyRandomNumber<uint32_t>();
    });
    ASSERT(!m_randomSource);
    m_randomSource.emplace(globalCounter++);
}

}

namespace WTF {

void printInternal(PrintStream& out, JSC::AbstractMacroAssemblerBase::StatusCondition condition)
{
    switch (condition) {
    case JSC::AbstractMacroAssemblerBase::Success:
        out.print("Success");
        return;
    case JSC::AbstractMacroAssemblerBase::Failure:
        out.print("Failure");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(ASSEMBLER)
