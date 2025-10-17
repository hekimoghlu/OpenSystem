/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#import "ApplePushServiceConnection.h"

#if HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)

#import <wtf/BlockPtr.h>
#import <wtf/WeakPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

@interface _WKAPSConnectionDelegate : NSObject<APSConnectionDelegate> {
    WeakPtr<WebPushD::ApplePushServiceConnection> _connection;
}
@end

@implementation _WKAPSConnectionDelegate

- (instancetype)initWithConnection:(WebPushD::ApplePushServiceConnection *)connection
{
    if ((self = [super init]))
        _connection = connection;
    return self;
}

- (void)connection:(APSConnection *)connection didReceivePublicToken:(NSData *)publicToken
{
    UNUSED_PARAM(connection);
    ASSERT(isMainRunLoop());

    if (_connection && publicToken.length)
        _connection->didReceivePublicToken(makeVector(publicToken));
}

- (void)connection:(APSConnection *)connection didReceiveIncomingMessage:(APSIncomingMessage *)message
{
    UNUSED_PARAM(connection);
    ASSERT(isMainRunLoop());

    if (_connection)
        _connection->didReceivePushMessage(message.topic, message.userInfo);
}

@end

namespace WebPushD {

ApplePushServiceConnection::ApplePushServiceConnection(const String& incomingPushServiceName)
{
    m_connection = adoptNS([[APSConnection alloc] initWithEnvironmentName:APSEnvironmentProduction namedDelegatePort:incomingPushServiceName queue:dispatch_get_main_queue()]);
    m_delegate = adoptNS([[_WKAPSConnectionDelegate alloc] initWithConnection:this]);
    [m_connection setDelegate:m_delegate.get()];
}

ApplePushServiceConnection::~ApplePushServiceConnection()
{
    [m_connection setDelegate:nil];
}

static RetainPtr<APSURLTokenInfo> makeTokenInfo(const String& topic, const Vector<uint8_t>& vapidPublicKey)
{
    return adoptNS([[APSURLTokenInfo alloc] initWithTopic:topic vapidPublicKey:toNSData(vapidPublicKey).get()]);
}

void ApplePushServiceConnection::subscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, SubscribeHandler&& subscribeHandler)
{
    ASSERT(isMainRunLoop());

    // Stash the completion handler away and look it up by id so that we can ensure it gets destructed on the main thread. If we move the handler and capture it in the Obj-C block, it might get destructed on a secondary thread since this completion block moves between different dispatch queues in the APS implementation.
    auto identifier = ++m_handlerIdentifier;
    m_subscribeHandlers.add(identifier, WTFMove(subscribeHandler));

    [m_connection requestURLTokenForInfo:makeTokenInfo(topic, vapidPublicKey).get() completion:makeBlockPtr([this, weakThis = WeakPtr { *this }, identifier] (APSURLToken *token, NSError *error) {
        if (!weakThis)
            return;

        auto handler = m_subscribeHandlers.take(identifier);
        handler(token.tokenURL, error);
    }).get()];
}

void ApplePushServiceConnection::unsubscribe(const String& topic, const Vector<uint8_t>& vapidPublicKey, UnsubscribeHandler&& unsubscribeHandler)
{
    ASSERT(isMainRunLoop());

    // See subscribe for why we stash the handler into a map.
    auto identifier = ++m_handlerIdentifier;
    m_unsubscribeHandlers.add(identifier, WTFMove(unsubscribeHandler));

    [m_connection invalidateURLTokenForInfo:makeTokenInfo(topic, vapidPublicKey).get() completion:makeBlockPtr([this, weakThis = WeakPtr { *this }, identifier] (BOOL success, NSError *error) {
        if (!weakThis)
            return;

        auto handler = m_unsubscribeHandlers.take(identifier);
        handler(success, error);
    }).get()];
}

Vector<String> ApplePushServiceConnection::enabledTopics()
{
    return makeVector<String>([m_connection enabledTopics]);
}

Vector<String> ApplePushServiceConnection::ignoredTopics()
{
    return makeVector<String>([m_connection ignoredTopics]);
}

Vector<String> ApplePushServiceConnection::opportunisticTopics()
{
    return makeVector<String>([m_connection opportunisticTopics]);
}

Vector<String> ApplePushServiceConnection::nonWakingTopics()
{
    return makeVector<String>([m_connection nonWakingTopics]);
}

void ApplePushServiceConnection::setEnabledTopics(Vector<String>&& topics)
{
    [m_connection _setEnabledTopics:createNSArray(WTFMove(topics)).get()];
}

void ApplePushServiceConnection::setIgnoredTopics(Vector<String>&& topics)
{
    [m_connection _setIgnoredTopics:createNSArray(WTFMove(topics)).get()];
}

void ApplePushServiceConnection::setOpportunisticTopics(Vector<String>&& topics)
{
    [m_connection _setOpportunisticTopics:createNSArray(WTFMove(topics)).get()];
}

void ApplePushServiceConnection::setNonWakingTopics(Vector<String>&& topics)
{
    [m_connection _setNonWakingTopics:createNSArray(WTFMove(topics)).get()];
}

void ApplePushServiceConnection::setTopicLists(TopicLists&& topicLists)
{
    [m_connection setEnabledTopics:createNSArray(WTFMove(topicLists.enabledTopics)).get() ignoredTopics:createNSArray(WTFMove(topicLists.ignoredTopics)).get() opportunisticTopics:createNSArray(WTFMove(topicLists.opportunisticTopics)).get() nonWakingTopics:createNSArray(WTFMove(topicLists.nonWakingTopics)).get()];
}

} // namespace WebPushD

#endif // HAVE(APPLE_PUSH_SERVICE_URL_TOKEN_SUPPORT)
