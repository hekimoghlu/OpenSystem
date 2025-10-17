/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#import "_WKWebPushDaemonConnectionInternal.h"

#import "APIWebPushMessage.h"
#import "APIWebPushSubscriptionData.h"
#import "WKError.h"
#import "WKSecurityOriginInternal.h"
#import "WebPushDaemonConnectionConfiguration.h"
#import "_WKNotificationDataInternal.h"
#import "_WKWebPushSubscriptionDataInternal.h"
#import <WebCore/ExceptionData.h>
#import <WebCore/PushPermissionState.h>
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>
#import <wtf/RetainPtr.h>
#import <wtf/StdLibExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKWebPushDaemonConnectionConfiguration

- (instancetype)init
{
    self = [super init];
    if (!self)
        return nil;

#if ENABLE(RELOCATABLE_WEBPUSHD)
    _machServiceName = @"com.apple.webkit.webpushd.relocatable.service";
#else
    _machServiceName = @"com.apple.webkit.webpushd.service";
#endif
    return self;
}

@end

@implementation _WKWebPushDaemonConnection

- (instancetype)initWithConfiguration:(_WKWebPushDaemonConnectionConfiguration *)configuration
{
    if (!(self = [super init]))
        return nil;

    WebKit::WebPushD::WebPushDaemonConnectionConfiguration connectionConfiguration {
        { },
        configuration.bundleIdentifierOverrideForTesting,
        configuration.partition,
        { },
    };

#if !USE(EXTENSIONKIT)
    auto hostAppAuditToken = configuration.hostApplicationAuditToken;
    Vector<uint8_t> hostAppAuditTokenData(sizeof(hostAppAuditToken));
    memcpySpan(hostAppAuditTokenData.mutableSpan(), asByteSpan(hostAppAuditToken));
    connectionConfiguration.hostAppAuditTokenData = WTFMove(hostAppAuditTokenData);
#endif

    API::Object::constructInWrapper<API::WebPushDaemonConnection>(self, configuration.machServiceName, WTFMove(connectionConfiguration));

    return self;
}

static _WKWebPushPermissionState toWKPermissionsState(WebCore::PushPermissionState state)
{
    switch (state) {
    case WebCore::PushPermissionState::Denied:
        return _WKWebPushPermissionStateDenied;
    case WebCore::PushPermissionState::Granted:
        return _WKWebPushPermissionStateGranted;
    case WebCore::PushPermissionState::Prompt:
        return _WKWebPushPermissionStatePrompt;
    }

    return _WKWebPushPermissionStateDenied;
}

- (void)getPushPermissionStateForOrigin:(NSURL *)originURL completionHandler:(void (^)(_WKWebPushPermissionState))completionHandler
{
    self._protectedConnection->getPushPermissionState(originURL, [completionHandlerCopy = makeBlockPtr(completionHandler)](auto result) {
        completionHandlerCopy(toWKPermissionsState(result));
    });
}

- (void)requestPushPermissionForOrigin:(NSURL *)originURL completionHandler:(void (^)(BOOL))completionHandler
{
    self._protectedConnection->requestPushPermission(originURL, [completionHandlerCopy = makeBlockPtr(completionHandler)] (bool result) {
        completionHandlerCopy(result);
    });
}

- (void)setAppBadge:(NSUInteger *)badge origin:(NSURL *)originURL
{
    std::optional<uint64_t> badgeValue = badge ? std::optional<uint64_t> { (uint64_t)badge } : std::nullopt;
    self._protectedConnection->setAppBadge(originURL, badgeValue);
}

- (void)subscribeToPushServiceForScope:(NSURL *)scopeURL applicationServerKey:(NSData *)applicationServerKey completionHandler:(void (^)(_WKWebPushSubscriptionData *, NSError *))completionHandler
{
    auto key = makeVector(applicationServerKey);
    self._protectedConnection->subscribeToPushService(scopeURL, WTFMove(key), [completionHandlerCopy = makeBlockPtr(completionHandler)] (auto result) {
        if (result)
            return completionHandlerCopy(wrapper(API::WebPushSubscriptionData::create(WTFMove(result.value()))).get(), nil);

        // FIXME: This error can be used to create DOMException; we may consider adding a new value to WKErrorCode for it.
        auto error = [NSError errorWithDomain:@"WKErrorDomain" code:WKErrorUnknown userInfo:@{ NSLocalizedDescriptionKey:result.error().message }];
        completionHandlerCopy(nil, error);
    });
}

- (void)unsubscribeFromPushServiceForScope:(NSURL *)scopeURL completionHandler:(void (^)(BOOL unsubscribed, NSError *))completionHandler
{
    self._protectedConnection->unsubscribeFromPushService(scopeURL, [completionHandlerCopy = makeBlockPtr(completionHandler)] (auto result) {
        if (result)
            return completionHandlerCopy(result.value(), nil);

        auto error = [NSError errorWithDomain:@"WKErrorDomain" code:WKErrorUnknown userInfo:@{ NSLocalizedDescriptionKey:result.error().message }];
        completionHandlerCopy(false, error);
    });
}

- (void)getSubscriptionForScope:(NSURL *)scopeURL completionHandler:(void (^)(_WKWebPushSubscriptionData *, NSError *))completionHandler
{
    self._protectedConnection->getPushSubscription(scopeURL, [completionHandlerCopy = makeBlockPtr(completionHandler)] (auto result) {
        if (result) {
            if (auto data = result.value())
                return completionHandlerCopy(wrapper(API::WebPushSubscriptionData::create(WTFMove(*data))).get(), nil);

            return completionHandlerCopy(nil, nil);
        }

        auto error = [NSError errorWithDomain:@"WKErrorDomain" code:WKErrorUnknown userInfo:@{ NSLocalizedDescriptionKey:result.error().message }];
        completionHandlerCopy(nil, error);
    });
}

- (void)getNextPendingPushMessage:(void (^)(_WKWebPushMessage *))completionHandler
{
    self._protectedConnection->getNextPendingPushMessage([completionHandlerCopy = makeBlockPtr(completionHandler)] (auto result) {
        if (!result)
            return completionHandlerCopy(nil);

        return completionHandlerCopy(wrapper(API::WebPushMessage::create(WTFMove(result.value()))).get());
    });
}


- (void)showNotification:(_WKNotificationData *)notificationData completionHandler:(void (^)())completionHandler
{
    self._protectedConnection->showNotification([notificationData _getCoreData], [completionHandlerCopy = makeBlockPtr(completionHandler)] () {
        completionHandlerCopy();
    });
}

- (void)getNotifications:(NSURL *)scopeURL tag:(NSString *)tag completionHandler:(void (^)(NSArray<_WKNotificationData *> *, NSError *))completionHandler
{
    self._protectedConnection->getNotifications(scopeURL, tag, [completionHandlerCopy = makeBlockPtr(completionHandler)] (auto result) {
        if (result) {
            NSMutableArray<_WKNotificationData *> *nsResult = [NSMutableArray arrayWithCapacity:result.value().size()];
            for (auto& data : result.value())
                [nsResult addObject:[[[_WKNotificationData alloc] _initWithCoreData:data] autorelease]];

            return completionHandlerCopy(nsResult, nil);
        }

        auto error = [NSError errorWithDomain:@"WKErrorDomain" code:WKErrorUnknown userInfo:@{ NSLocalizedDescriptionKey:result.error().message }];
        completionHandlerCopy(nil, error);
    });
}

- (void)cancelNotification:(NSURL *)securityOriginURL uuid:(NSUUID *)notificationIdentifier
{
    // UUID::fromNSUUID only fails if the passed in NSUUID is nil, which would be a crash-worthy misuse of this API
    self._protectedConnection->cancelNotification(securityOriginURL, *WTF::UUID::fromNSUUID(notificationIdentifier));
}

- (API::Object&)_apiObject
{
    return *_connection;
}

- (Ref<API::WebPushDaemonConnection>)_protectedConnection
{
    return *_connection;
}

@end
