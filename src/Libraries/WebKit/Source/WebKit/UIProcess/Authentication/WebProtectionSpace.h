/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#ifndef WebProtectionSpace_h
#define WebProtectionSpace_h

#include "APIObject.h"
#include <WebCore/ProtectionSpace.h>

namespace WebKit {

class WebProtectionSpace : public API::ObjectImpl<API::Object::Type::ProtectionSpace> {
public:
    static Ref<WebProtectionSpace> create(const WebCore::ProtectionSpace& protectionSpace)
    {
        return adoptRef(*new WebProtectionSpace(protectionSpace));
    }
    
    const String& protocol() const;
    const String& host() const;
    int port() const;
    const String& realm() const;
    bool isProxy() const;
    WebCore::ProtectionSpace::ServerType serverType() const;
    bool receivesCredentialSecurely() const;
    WebCore::ProtectionSpace::AuthenticationScheme authenticationScheme() const;

    const WebCore::ProtectionSpace& protectionSpace() const { return m_coreProtectionSpace; }

private:
    explicit WebProtectionSpace(const WebCore::ProtectionSpace&);

    WebCore::ProtectionSpace m_coreProtectionSpace;
};

} // namespace WebKit

#endif // WebProtectionSpace_h
