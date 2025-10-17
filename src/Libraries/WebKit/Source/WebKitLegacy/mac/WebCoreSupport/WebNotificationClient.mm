/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#import "WebNotificationClient.h"

#if ENABLE(NOTIFICATIONS)

#import "WebDelegateImplementationCaching.h"
#import "WebNotificationInternal.h"
#import "WebPreferencesPrivate.h"
#import "WebSecurityOriginInternal.h"
#import "WebUIDelegatePrivate.h"
#import "WebViewInternal.h"
#import <WebCore/ScriptExecutionContext.h>
#import <WebCore/SecurityOrigin.h>
#import <wtf/BlockObjCExceptions.h>
#import <wtf/CompletionHandler.h>
#import <wtf/Scope.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/VectorCocoa.h>

using namespace WebCore;

@interface WebNotificationPolicyListener : NSObject <WebAllowDenyPolicyListener>
{
    NotificationClient::PermissionHandler _permissionHandler;
}
- (id)initWithPermissionHandler:(NotificationClient::PermissionHandler&&)permissionHandler;
@end

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebNotificationClient);

WebNotificationClient::WebNotificationClient(WebView *webView)
    : m_webView(webView)
{
}

bool WebNotificationClient::show(ScriptExecutionContext&, NotificationData&& notification, RefPtr<NotificationResources>&&, CompletionHandler<void()>&& callback)
{
    auto scope = makeScopeExit([&callback] { callback(); });

    if (![m_webView _notificationProvider])
        return false;

    auto notificationID = notification.notificationID;
    RetainPtr<WebNotification> webNotification = adoptNS([[WebNotification alloc] initWithCoreNotification:WTFMove(notification)]);
    m_notificationMap.set(notificationID, webNotification);

    [[m_webView _notificationProvider] showNotification:webNotification.get() fromWebView:m_webView];
    return true;
}

void WebNotificationClient::cancel(NotificationData&& notification)
{
    WebNotification *webNotification = m_notificationMap.get(notification.notificationID).get();
    if (!webNotification)
        return;

    [[m_webView _notificationProvider] cancelNotification:webNotification];
}

void WebNotificationClient::notificationObjectDestroyed(NotificationData&& notification)
{
    RetainPtr<WebNotification> webNotification = m_notificationMap.take(notification.notificationID);
    if (!webNotification)
        return;

    [[m_webView _notificationProvider] notificationDestroyed:webNotification.get()];
}

void WebNotificationClient::notificationControllerDestroyed()
{
    delete this;
}

void WebNotificationClient::clearNotificationPermissionState()
{
    m_notificationPermissionRequesters.clear();
}

void WebNotificationClient::requestPermission(ScriptExecutionContext& context, WebNotificationPolicyListener *listener)
{
    SEL selector = @selector(webView:decidePolicyForNotificationRequestFromOrigin:listener:);
    if (![[m_webView UIDelegate] respondsToSelector:selector])
        return;

    m_everRequestedPermission = true;

    auto webOrigin = adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:context.securityOrigin()]);

    // Add origin to list of origins that have requested permission to use the Notifications API.
    m_notificationPermissionRequesters.add(context.securityOrigin()->data());
    
    CallUIDelegate(m_webView, selector, webOrigin.get(), listener);
}

void WebNotificationClient::requestPermission(ScriptExecutionContext& context, PermissionHandler&& permissionHandler)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS
    auto listener = adoptNS([[WebNotificationPolicyListener alloc] initWithPermissionHandler:WTFMove(permissionHandler)]);
    requestPermission(context, listener.get());
    END_BLOCK_OBJC_EXCEPTIONS
}

NotificationClient::Permission WebNotificationClient::checkPermission(ScriptExecutionContext* context)
{
    if (!context || !context->isDocument())
        return NotificationClient::Permission::Denied;
    if (![[m_webView preferences] notificationsEnabled])
        return NotificationClient::Permission::Denied;
    auto webOrigin = adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:context->securityOrigin()]);
    WebNotificationPermission permission = [[m_webView _notificationProvider] policyForOrigin:webOrigin.get()];

    // To reduce fingerprinting, if the origin has not requested permission to use the
    // Notifications API, and the permission state is "denied", return "default" instead.
    if (permission == WebNotificationPermissionDenied && !m_notificationPermissionRequesters.contains(context->securityOrigin()->data()))
        return NotificationClient::Permission::Default;

    switch (permission) {
        case WebNotificationPermissionAllowed:
            return NotificationClient::Permission::Granted;
        case WebNotificationPermissionDenied:
            return NotificationClient::Permission::Denied;
        case WebNotificationPermissionNotAllowed:
            return NotificationClient::Permission::Default;
        default:
            return NotificationClient::Permission::Default;
    }
}

@implementation WebNotificationPolicyListener

- (id)initWithPermissionHandler:(NotificationClient::PermissionHandler&&)permissionHandler
{
    if (!(self = [super init]))
        return nil;

    _permissionHandler = WTFMove(permissionHandler);
    return self;
}

- (void)allow
{
    if (_permissionHandler)
        _permissionHandler(NotificationClient::Permission::Granted);
}

- (void)deny
{
    if (_permissionHandler)
        _permissionHandler(NotificationClient::Permission::Denied);
}

#if PLATFORM(IOS_FAMILY)
- (void)denyOnlyThisRequest NO_RETURN_DUE_TO_ASSERT
{
    ASSERT_NOT_REACHED();
}

- (BOOL)shouldClearCache NO_RETURN_DUE_TO_ASSERT
{
    ASSERT_NOT_REACHED();
    return NO;
}
#endif

@end

#endif
