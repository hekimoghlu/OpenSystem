/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#import "config.h"
#import "MockPushServiceConnection.h"

#import <wtf/text/Base64.h>

namespace WebPushD {
using namespace WebCore::PushCrypto;

MockPushServiceConnection::MockPushServiceConnection()
{
    didReceivePublicToken(Vector<uint8_t> { 'a', 'b', 'c' });
}

MockPushServiceConnection::~MockPushServiceConnection() = default;

ClientKeys MockPushServiceConnection::generateClientKeys()
{
    // Example values from RFC8291 Section 5.
    auto publicKey = base64URLDecode("BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4"_s).value();
    auto privateKey = base64URLDecode("q1dXpw3UpT5VOmu_cf_v6ih07Aems3njxI-JWgLcM94"_s).value();
    auto secret = base64URLDecode("BTBZMqHH6r4Tts7J_aSIgg"_s).value();

    return ClientKeys { P256DHKeyPair { WTFMove(publicKey), WTFMove(privateKey) }, WTFMove(secret) };
}

void MockPushServiceConnection::subscribe(const String&, const Vector<uint8_t>& vapidPublicKey, SubscribeHandler&& handler)
{
    auto alwaysRejectedKey = base64URLDecode("BEAxaUMo1s8tjORxJfnSSvWhYb4u51kg1hWT2s_9gpV7Zxar1pF_2BQ8AncuAdS2BoLhN4qaxzBy2CwHE8BBzWg"_s).value();
    if (vapidPublicKey == alwaysRejectedKey) {
        handler({ }, [NSError errorWithDomain:@"WebPush" code:-1 userInfo:nil]);
        return;
    }

    handler(@"https://webkit.org/push", nil);
}

void MockPushServiceConnection::unsubscribe(const String&, const Vector<uint8_t>&, UnsubscribeHandler&& handler)
{
    handler(true, nil);
}

void MockPushServiceConnection::setTopicLists(TopicLists&& topics)
{
    setEnabledTopics(WTFMove(topics.enabledTopics));
    setIgnoredTopics(WTFMove(topics.ignoredTopics));
    setOpportunisticTopics(WTFMove(topics.opportunisticTopics));
    setNonWakingTopics(WTFMove(topics.nonWakingTopics));
}

void MockPushServiceConnection::setPublicTokenForTesting(Vector<uint8_t>&& token)
{
    didReceivePublicToken(WTFMove(token));
}

} // namespace WebPushD
