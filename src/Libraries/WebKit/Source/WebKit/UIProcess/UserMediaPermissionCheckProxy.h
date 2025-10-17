/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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

#include "APIObject.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/MediaConstraints.h>
#include <wtf/CompletionHandler.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SecurityOrigin;
}

namespace WebKit {

class UserMediaPermissionCheckProxy : public API::ObjectImpl<API::Object::Type::UserMediaPermissionCheck> {
public:
    enum class PermissionInfo { Error, Unknown, Granted };
    using CompletionHandler = WTF::CompletionHandler<void(PermissionInfo)>;

    static Ref<UserMediaPermissionCheckProxy> create(WebCore::FrameIdentifier frameID, CompletionHandler&& handler, Ref<WebCore::SecurityOrigin>&& userMediaDocumentOrigin, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin)
    {
        return adoptRef(*new UserMediaPermissionCheckProxy(frameID, WTFMove(handler), WTFMove(userMediaDocumentOrigin), WTFMove(topLevelDocumentOrigin)));
    }

    void deny() { setUserMediaAccessInfo(false); }
    void setUserMediaAccessInfo(bool);
    void invalidate() { complete(PermissionInfo::Error); }

    WebCore::FrameIdentifier frameID() const { return m_frameID; }
    WebCore::SecurityOrigin& userMediaDocumentSecurityOrigin() { return m_userMediaDocumentSecurityOrigin.get(); }
    WebCore::SecurityOrigin& topLevelDocumentSecurityOrigin() { return m_topLevelDocumentSecurityOrigin.get(); }
    
private:
    UserMediaPermissionCheckProxy(WebCore::FrameIdentifier, CompletionHandler&&, Ref<WebCore::SecurityOrigin>&& userMediaDocumentOrigin, Ref<WebCore::SecurityOrigin>&& topLevelDocumentOrigin);
    ~UserMediaPermissionCheckProxy();

    void complete(PermissionInfo);
    
    WebCore::FrameIdentifier m_frameID;
    CompletionHandler m_completionHandler;
    Ref<WebCore::SecurityOrigin> m_userMediaDocumentSecurityOrigin;
    Ref<WebCore::SecurityOrigin> m_topLevelDocumentSecurityOrigin;
};

} // namespace WebKit
