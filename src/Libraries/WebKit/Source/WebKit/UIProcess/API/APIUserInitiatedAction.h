/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
#include <wtf/UUID.h>

namespace API {

class UserInitiatedAction final : public API::ObjectImpl<API::Object::Type::UserInitiatedAction> {
public:
    static Ref<UserInitiatedAction> create()
    {
        return adoptRef(*new UserInitiatedAction);
    }

    UserInitiatedAction() = default;
    virtual ~UserInitiatedAction() = default;

    void setConsumed() { m_consumed = true; }
    bool consumed() const { return m_consumed; }

    void setAuthorizationToken(WTF::UUID authorizationToken) { m_authorizationToken = authorizationToken; }
    std::optional<WTF::UUID> authorizationToken() const { return m_authorizationToken; }

private:
    bool m_consumed { false };
    std::optional<WTF::UUID> m_authorizationToken;
};

} // namespace API
