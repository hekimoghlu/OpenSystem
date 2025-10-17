/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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
#include "WebProtectionSpace.h"

#include <WebCore/SharedBuffer.h>

namespace WebKit {

WebProtectionSpace::WebProtectionSpace(const WebCore::ProtectionSpace& coreProtectionSpace)
    : m_coreProtectionSpace(coreProtectionSpace)
{
}

const String& WebProtectionSpace::host() const
{
    return m_coreProtectionSpace.host();
}

int WebProtectionSpace::port() const
{
    return m_coreProtectionSpace.port();
}

const String& WebProtectionSpace::realm() const
{
    return m_coreProtectionSpace.realm();
}

bool WebProtectionSpace::isProxy() const
{
    return m_coreProtectionSpace.isProxy();
}

WebCore::ProtectionSpace::ServerType WebProtectionSpace::serverType() const
{
    return m_coreProtectionSpace.serverType();
}

bool WebProtectionSpace::receivesCredentialSecurely() const
{
    return m_coreProtectionSpace.receivesCredentialSecurely();
}

WebCore::ProtectionSpace::AuthenticationScheme WebProtectionSpace::authenticationScheme() const
{
    return m_coreProtectionSpace.authenticationScheme();
}

} // namespace WebKit
