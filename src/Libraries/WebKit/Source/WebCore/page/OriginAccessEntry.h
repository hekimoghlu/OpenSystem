/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

class SecurityOrigin;

class OriginAccessEntry {
public:
    enum SubdomainSetting {
        AllowSubdomains,
        DisallowSubdomains
    };

    enum IPAddressSetting {
        TreatIPAddressAsDomain,
        TreatIPAddressAsIPAddress
    };

    // If host is empty string and SubdomainSetting is AllowSubdomains, the entry will match all domains in the specified protocol.
    OriginAccessEntry(const String& protocol, const String& host, SubdomainSetting, IPAddressSetting);
    bool matchesOrigin(const SecurityOrigin&) const;

    const String& protocol() const { return m_protocol; }
    const String& host() const { return m_host; }
    SubdomainSetting subdomainSettings() const { return m_subdomainSettings; }
    IPAddressSetting ipAddressSettings() const { return m_ipAddressSettings; }

private:
    String m_protocol;
    String m_host;
    SubdomainSetting m_subdomainSettings;
    IPAddressSetting m_ipAddressSettings;
    bool m_hostIsIPAddress;
};

inline bool operator==(const OriginAccessEntry& a, const OriginAccessEntry& b)
{
    return equalIgnoringASCIICase(a.protocol(), b.protocol())
        && equalIgnoringASCIICase(a.host(), b.host())
        && a.subdomainSettings() == b.subdomainSettings()
        && a.ipAddressSettings() == b.ipAddressSettings();
}

} // namespace WebCore
