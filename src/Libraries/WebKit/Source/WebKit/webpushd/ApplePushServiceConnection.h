/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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

#if HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)

#include "ApplePushServiceSPI.h"
#include "PushServiceConnection.h"
#include <wtf/HashMap.h>

namespace WebPushD {

class ApplePushServiceConnection final : public PushServiceConnection {
public:
    static Ref<ApplePushServiceConnection> create(const String& incomingPushServiceName)
    {
        return adoptRef(*new ApplePushServiceConnection(incomingPushServiceName));
    }

    ~ApplePushServiceConnection();

    void subscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, SubscribeHandler&&) final;
    void unsubscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, UnsubscribeHandler&&) final;

    Vector<String> enabledTopics() override;
    Vector<String> ignoredTopics() override;
    Vector<String> opportunisticTopics() override;
    Vector<String> nonWakingTopics() override;

    void setEnabledTopics(Vector<String>&&) override;
    void setIgnoredTopics(Vector<String>&&) override;
    void setOpportunisticTopics(Vector<String>&&) override;
    void setNonWakingTopics(Vector<String>&&) override;

    void setTopicLists(TopicLists&&) override;

private:
    ApplePushServiceConnection(const String& incomingPushServiceName);

    RetainPtr<APSConnection> m_connection;
    RetainPtr<id<APSConnectionDelegate>> m_delegate;
    unsigned m_handlerIdentifier { 0 };
    HashMap<unsigned, SubscribeHandler> m_subscribeHandlers;
    HashMap<unsigned, UnsubscribeHandler> m_unsubscribeHandlers;
};

} // namespace WebPushD

#endif // HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)

