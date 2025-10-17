/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include <WebCore/PushSubscriptionData.h>

namespace API {

class WebPushSubscriptionData final : public ObjectImpl<Object::Type::WebPushSubscriptionData> {
public:
    static Ref<WebPushSubscriptionData> create(WebCore::PushSubscriptionData&& data)
    {
        return adoptRef(*new WebPushSubscriptionData(WTFMove(data)));
    }

    explicit WebPushSubscriptionData(WebCore::PushSubscriptionData&& data)
        : m_data(WTFMove(data)) { }

    WTF::URL endpoint() const { return WTF::URL { m_data.endpoint }; }
    Vector<uint8_t> applicationServerKey() const { return m_data.serverVAPIDPublicKey; }
    Vector<uint8_t> clientECDHPublicKey() const { return m_data.clientECDHPublicKey; }
    Vector<uint8_t> sharedAuthenticationSecret() const { return m_data.sharedAuthenticationSecret; }

private:
    const WebCore::PushSubscriptionData m_data;
};

} // namespace API
