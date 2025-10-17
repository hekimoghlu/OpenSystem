/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#include "PushServiceConnection.h"

namespace WebPushD {

class MockPushServiceConnection final : public PushServiceConnection {
public:
    static Ref<MockPushServiceConnection> create()
    {
        return adoptRef(*new MockPushServiceConnection());
    }

    ~MockPushServiceConnection();

    WebCore::PushCrypto::ClientKeys generateClientKeys() final;

    void subscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, SubscribeHandler&&) override;
    void unsubscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, UnsubscribeHandler&&) override;

    Vector<String> enabledTopics() override { return m_enabledTopics; }
    Vector<String> ignoredTopics() override { return m_ignoredTopics; }
    Vector<String> opportunisticTopics() override { return m_opportunisticTopics; }
    Vector<String> nonWakingTopics() override { return m_nonWakingTopics; }

    void setEnabledTopics(Vector<String>&& enabledTopics) override { m_enabledTopics = enabledTopics; }
    void setIgnoredTopics(Vector<String>&& ignoredTopics) override { m_ignoredTopics = ignoredTopics; }
    void setOpportunisticTopics(Vector<String>&& opportunisticTopics) override { m_opportunisticTopics = opportunisticTopics; }
    void setNonWakingTopics(Vector<String>&& nonWakingTopics) override { m_nonWakingTopics = nonWakingTopics; }

    void setTopicLists(TopicLists&&) override;

    void setPublicTokenForTesting(Vector<uint8_t>&&) override;

private:
    MockPushServiceConnection();

    Vector<String> m_enabledTopics;
    Vector<String> m_ignoredTopics;
    Vector<String> m_opportunisticTopics;
    Vector<String> m_nonWakingTopics;
};

} // namespace WebPushD
