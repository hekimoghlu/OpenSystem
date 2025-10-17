/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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

#include <WebCore/PushMessageCrypto.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Deque.h>
#include <wtf/Function.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSError;
OBJC_CLASS NSString;
OBJC_CLASS NSDictionary;

namespace WebPushD {

class PushServiceConnection : public RefCountedAndCanMakeWeakPtr<PushServiceConnection> {
    WTF_MAKE_TZONE_ALLOCATED(PushServiceConnection);
public:
    using IncomingPushMessageHandler = Function<void(NSString *, NSDictionary *)>;

    virtual ~PushServiceConnection() = default;

    virtual WebCore::PushCrypto::ClientKeys generateClientKeys();

    using SubscribeHandler = CompletionHandler<void(NSString *, NSError *)>;
    virtual void subscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, SubscribeHandler&&) = 0;

    using UnsubscribeHandler = CompletionHandler<void(bool, NSError *)>;
    virtual void unsubscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, UnsubscribeHandler&&) = 0;

    virtual Vector<String> enabledTopics() = 0;
    virtual Vector<String> ignoredTopics() = 0;
    virtual Vector<String> opportunisticTopics() = 0;
    virtual Vector<String> nonWakingTopics() = 0;

    virtual void setEnabledTopics(Vector<String>&&) = 0;
    virtual void setIgnoredTopics(Vector<String>&&) = 0;
    virtual void setOpportunisticTopics(Vector<String>&&) = 0;
    virtual void setNonWakingTopics(Vector<String>&&) = 0;

    struct TopicLists {
        Vector<String> enabledTopics;
        Vector<String> ignoredTopics;
        Vector<String> opportunisticTopics;
        Vector<String> nonWakingTopics;
    };
    virtual void setTopicLists(TopicLists&&) = 0;

    void startListeningForPublicToken(Function<void(Vector<uint8_t>&&)>&&);
    void didReceivePublicToken(Vector<uint8_t>&&);
    virtual void setPublicTokenForTesting(Vector<uint8_t>&&);

    void startListeningForPushMessages(IncomingPushMessageHandler&&);
    void didReceivePushMessage(NSString *topic, NSDictionary *userInfo);

protected:
    PushServiceConnection() = default;

private:
    Function<void(Vector<uint8_t>&&)> m_publicTokenChangeHandler;
    Vector<uint8_t> m_pendingPublicToken;
    IncomingPushMessageHandler m_incomingPushMessageHandler;
    Deque<std::pair<RetainPtr<NSString>, RetainPtr<NSDictionary>>> m_pendingPushes;
};

} // namespace WebPushD
