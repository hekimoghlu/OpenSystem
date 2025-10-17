/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
#include "WebDriverService.h"

#include <wtf/text/StringToIntegerConversion.h>

namespace WebDriver {

static bool parseVersion(const String& version, uint64_t& major, uint64_t& minor, uint64_t& micro)
{
    major = minor = micro = 0;

    Vector<String> tokens = version.split('.');
    switch (tokens.size()) {
    case 3: {
        auto parsedMicro = parseIntegerAllowingTrailingJunk<uint64_t>(tokens[2]);
        if (!parsedMicro)
            return false;
        micro = *parsedMicro;
    }
        FALLTHROUGH;
    case 2: {
        auto parsedMinor = parseIntegerAllowingTrailingJunk<uint64_t>(tokens[1]);
        if (!parsedMinor)
            return false;
        minor = *parsedMinor;
    }
        FALLTHROUGH;
    case 1: {
        auto parsedMajor = parseIntegerAllowingTrailingJunk<uint64_t>(tokens[0]);
        if (!parsedMajor)
            return false;
        major = *parsedMajor;
    }
        break;
    default:
        return false;
    }

    return true;
}

bool WebDriverService::platformCompareBrowserVersions(const String& requiredVersion, const String& proposedVersion)
{
    // We require clients to use format major.minor.micro as version string.
    uint64_t requiredMajor, requiredMinor, requiredMicro;
    if (!parseVersion(requiredVersion, requiredMajor, requiredMinor, requiredMicro))
        return false;

    uint64_t proposedMajor, proposedMinor, proposedMicro;
    if (!parseVersion(proposedVersion, proposedMajor, proposedMinor, proposedMicro))
        return false;

    return proposedMajor > requiredMajor
        || (proposedMajor == requiredMajor && proposedMinor > requiredMinor)
        || (proposedMajor == requiredMajor && proposedMinor == requiredMinor && proposedMicro >= requiredMicro);
}

bool WebDriverService::platformSupportProxyType(const String&) const
{
    return true;
}

bool WebDriverService::platformSupportBidi() const
{
#if ENABLE(WEBDRIVER_BIDI)
    return true;
#else
    return false;
#endif
}

} // namespace WebDriver
