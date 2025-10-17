/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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

#include <wtf/Assertions.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class UserAgentQuirks {
public:
    enum UserAgentQuirk {
        NeedsChromeBrowser,
        NeedsFirefoxBrowser,
        NeedsMacintoshPlatform,
        NeedsUnbrandedUserAgent,

        NumUserAgentQuirks
    };

    UserAgentQuirks()
        : m_quirks(0)
    {
        static_assert(sizeof(m_quirks) * 8 >= NumUserAgentQuirks, "not enough room for quirks");
    }

    void add(UserAgentQuirk quirk)
    {
        ASSERT(quirk >= 0);
        ASSERT_WITH_SECURITY_IMPLICATION(quirk < NumUserAgentQuirks);

        m_quirks |= (1 << quirk);
    }

    bool contains(UserAgentQuirk quirk) const
    {
        return m_quirks & (1 << quirk);
    }

    bool isEmpty() const { return !m_quirks; }

    static UserAgentQuirks quirksForURL(const URL&);

    static String stringForQuirk(UserAgentQuirk);

private:
    uint32_t m_quirks;
};

}
