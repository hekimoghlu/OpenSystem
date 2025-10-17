/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#include "OriginAccessEntry.h"

#include "SecurityOrigin.h"

namespace WebCore {
    
OriginAccessEntry::OriginAccessEntry(const String& protocol, const String& host, SubdomainSetting subdomainSetting, IPAddressSetting ipAddressSetting)
    : m_protocol(protocol.convertToASCIILowercase())
    , m_host(host.convertToASCIILowercase())
    , m_subdomainSettings(subdomainSetting)
    , m_ipAddressSettings(ipAddressSetting)
    , m_hostIsIPAddress(URL::hostIsIPAddress(m_host))
{
    ASSERT(subdomainSetting == AllowSubdomains || subdomainSetting == DisallowSubdomains);
}

bool OriginAccessEntry::matchesOrigin(const SecurityOrigin& origin) const
{
    ASSERT(origin.host() == origin.host().convertToASCIILowercase());
    ASSERT(origin.protocol() == origin.protocol().convertToASCIILowercase());

    if (m_protocol != origin.protocol())
        return false;
    
    // Special case: Include subdomains and empty host means "all hosts, including ip addresses".
    if (m_subdomainSettings == AllowSubdomains && m_host.isEmpty())
        return true;
    
    // Exact match.
    if (m_host == origin.host())
        return true;
    
    // Otherwise we can only match if we're matching subdomains.
    if (m_subdomainSettings == DisallowSubdomains)
        return false;

    // IP addresses are not domains: https://url.spec.whatwg.org/#concept-domain
    // Don't try to do subdomain matching on IP addresses.
    if (m_ipAddressSettings == TreatIPAddressAsIPAddress && (m_hostIsIPAddress || URL::hostIsIPAddress(origin.host())))
        return false;
    
    // Match subdomains.
    if (origin.host().length() > m_host.length() && origin.host()[origin.host().length() - m_host.length() - 1] == '.' && origin.host().endsWith(m_host))
        return true;
    
    return false;
}
    
} // namespace WebCore
