/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

#include "APIContentWorld.h"
#include "APIObject.h"
#include "UserScriptIdentifier.h"
#include <WebCore/UserScript.h>
#include <wtf/Identified.h>

namespace API {

class UserScript final : public ObjectImpl<Object::Type::UserScript>, public Identified<WebKit::UserScriptIdentifier> {
public:
    static Ref<UserScript> create(WebCore::UserScript&& userScript, API::ContentWorld& world)
    {
        return adoptRef(*new UserScript(WTFMove(userScript), world));
    }

    UserScript(WebCore::UserScript, API::ContentWorld&);

    WebCore::UserScript& userScript() { return m_userScript; }
    const WebCore::UserScript& userScript() const { return m_userScript; }

    ContentWorld& contentWorld() { return m_world; }
    const ContentWorld& contentWorld() const { return m_world; }
    
private:
    WebCore::UserScript m_userScript;
    Ref<ContentWorld> m_world;
};

} // namespace API
