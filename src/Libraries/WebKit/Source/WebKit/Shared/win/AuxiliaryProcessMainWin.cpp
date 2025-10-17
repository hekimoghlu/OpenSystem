/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include "AuxiliaryProcessMain.h"

#include <JavaScriptCore/ExecutableAllocator.h>
#include <cstring>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebKit {

AuxiliaryProcessMainCommon::AuxiliaryProcessMainCommon() { }

bool AuxiliaryProcessMainCommon::parseCommandLine(int argc, char** argv)
{
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-clientIdentifier") && i + 1 < argc)
            m_parameters.connectionIdentifier = IPC::Connection::Identifier { reinterpret_cast<HANDLE>(parseIntegerAllowingTrailingJunk<uint64_t>(StringView::fromLatin1(argv[++i])).value_or(0)) };
        else if (!strcmp(argv[i], "-processIdentifier") && i + 1 < argc)
            m_parameters.processIdentifier = ObjectIdentifier<WebCore::ProcessIdentifierType>(parseIntegerAllowingTrailingJunk<uint64_t>(StringView::fromLatin1(argv[++i])).value_or(0));
        else if (!strcmp(argv[i], "-configure-jsc-for-testing"))
            JSC::Config::configureForTesting();
        else if (!strcmp(argv[i], "-disable-jit"))
            JSC::ExecutableAllocator::disableJIT();
    }
    return true;
}

} // namespace WebKit
