/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#include "UserStyleSheetIdentifier.h"
#include <WebCore/UserStyleSheet.h>
#include <wtf/Identified.h>

namespace API {

class UserStyleSheet final : public ObjectImpl<Object::Type::UserStyleSheet>, public Identified<WebKit::UserStyleSheetIdentifier> {
public:
    static Ref<UserStyleSheet> create(WebCore::UserStyleSheet userStyleSheet, API::ContentWorld& world)
    {
        return adoptRef(*new UserStyleSheet(WTFMove(userStyleSheet), world));
    }

    UserStyleSheet(WebCore::UserStyleSheet, API::ContentWorld&);

    const WebCore::UserStyleSheet& userStyleSheet() const { return m_userStyleSheet; }

    ContentWorld& contentWorld() { return m_world; }
    const ContentWorld& contentWorld() const { return m_world; }

private:
    WebCore::UserStyleSheet m_userStyleSheet;
    Ref<ContentWorld> m_world;
};

} // namespace API
