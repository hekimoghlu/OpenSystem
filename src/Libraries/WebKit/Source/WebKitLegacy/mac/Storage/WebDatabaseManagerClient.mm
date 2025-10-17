/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#import "WebDatabaseManagerClient.h"

#import "WebDatabaseManagerPrivate.h"
#import "WebSecurityOriginInternal.h"
#import <WebCore/DatabaseTracker.h>
#import <WebCore/SecurityOrigin.h>
#import <wtf/MainThread.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WebCoreThread.h>
#endif

using namespace WebCore;

#if PLATFORM(IOS_FAMILY)
static const CFStringRef WebDatabaseOriginWasAddedNotification = CFSTR("com.apple.MobileSafariSettings.WebDatabaseOriginWasAddedNotification");
static const CFStringRef WebDatabaseWasDeletedNotification = CFSTR("com.apple.MobileSafariSettings.WebDatabaseWasDeletedNotification");
static const CFStringRef WebDatabaseOriginWasDeletedNotification = CFSTR("com.apple.MobileSafariSettings.WebDatabaseOriginWasDeletedNotification");
#endif

namespace WebKit {

WebDatabaseManagerClient& WebDatabaseManagerClient::sharedWebDatabaseManagerClient()
{
    static NeverDestroyed<WebDatabaseManagerClient> sharedClient;
    return sharedClient;
}

#if PLATFORM(IOS_FAMILY)
static void onNewDatabaseOriginAdded(CFNotificationCenterRef, void* observer, CFStringRef, const void*, CFDictionaryRef)
{
    ASSERT(observer);
    
    WebDatabaseManagerClient* client = reinterpret_cast<WebDatabaseManagerClient*>(observer);
    client->newDatabaseOriginWasAdded();
}

static void onDatabaseDeleted(CFNotificationCenterRef, void* observer, CFStringRef, const void*, CFDictionaryRef)
{
    ASSERT(observer);
    
    WebDatabaseManagerClient* client = reinterpret_cast<WebDatabaseManagerClient*>(observer);
    client->databaseWasDeleted();
}

static void onDatabaseOriginDeleted(CFNotificationCenterRef, void* observer, CFStringRef, const void*, CFDictionaryRef)
{
    ASSERT(observer);
    
    WebDatabaseManagerClient* client = reinterpret_cast<WebDatabaseManagerClient*>(observer);
    client->databaseOriginWasDeleted();
}
#endif

WebDatabaseManagerClient::WebDatabaseManagerClient()
{
#if PLATFORM(IOS_FAMILY)
    CFNotificationCenterRef center = CFNotificationCenterGetDarwinNotifyCenter(); 
    CFNotificationCenterAddObserver(center, this, onNewDatabaseOriginAdded, WebDatabaseOriginWasAddedNotification, 0, CFNotificationSuspensionBehaviorDeliverImmediately);
    CFNotificationCenterAddObserver(center, this, onDatabaseDeleted, WebDatabaseWasDeletedNotification, 0, CFNotificationSuspensionBehaviorDeliverImmediately);
    CFNotificationCenterAddObserver(center, this, onDatabaseOriginDeleted, WebDatabaseOriginWasDeletedNotification, 0, CFNotificationSuspensionBehaviorDeliverImmediately);    
#endif
}

WebDatabaseManagerClient::~WebDatabaseManagerClient()
{
#if PLATFORM(IOS_FAMILY)
    CFNotificationCenterRemoveObserver(CFNotificationCenterGetDarwinNotifyCenter(), this, 0, 0);
#endif
}

class DidModifyOriginData {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DidModifyOriginData);
public:
    static void dispatchToMainThread(WebDatabaseManagerClient& client, const SecurityOriginData& origin)
    {
        auto context = makeUnique<DidModifyOriginData>(client, origin);
        callOnMainThread([context = WTFMove(context)] {
            context->client.dispatchDidModifyOrigin(context->origin);
        });
    }

    DidModifyOriginData(WebDatabaseManagerClient& client, const SecurityOriginData& origin)
        : client(client)
        , origin(origin.isolatedCopy())
    {
    }

private:
    WebDatabaseManagerClient& client;
    SecurityOriginData origin;
};

void WebDatabaseManagerClient::dispatchDidModifyOrigin(const SecurityOriginData& origin)
{
    if (!isMainThread()) {
        DidModifyOriginData::dispatchToMainThread(*this, origin);
        return;
    }

    auto webSecurityOrigin = adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:origin.securityOrigin().ptr()]);

    [[NSNotificationCenter defaultCenter] postNotificationName:WebDatabaseDidModifyOriginNotification object:webSecurityOrigin.get()];
}

void WebDatabaseManagerClient::dispatchDidModifyDatabase(const SecurityOriginData& origin, const String& databaseIdentifier)
{
    if (!isMainThread()) {
        DidModifyOriginData::dispatchToMainThread(*this, origin);
        return;
    }

    auto webSecurityOrigin = adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:origin.securityOrigin().ptr()]);
    auto userInfo = adoptNS([[NSDictionary alloc] initWithObjectsAndKeys:(NSString *)databaseIdentifier, WebDatabaseIdentifierKey, nil]);
    [[NSNotificationCenter defaultCenter] postNotificationName:WebDatabaseDidModifyDatabaseNotification object:webSecurityOrigin.get() userInfo:userInfo.get()];
}

#if PLATFORM(IOS_FAMILY)

void WebDatabaseManagerClient::dispatchDidAddNewOrigin()
{    
    m_isHandlingNewDatabaseOriginNotification = true;
    // Send a notification to all apps that a new origin has been added, so other apps with opened database can refresh their origin maps.
    CFNotificationCenterPostNotification(CFNotificationCenterGetDarwinNotifyCenter(), WebDatabaseOriginWasAddedNotification, nullptr, nullptr, true);
}

void WebDatabaseManagerClient::dispatchDidDeleteDatabase()
{
    m_isHandlingDeleteDatabaseNotification = true;
    // Send a notification to all apps that a database has been deleted, so other apps with the deleted database open will close it properly.
    CFNotificationCenterPostNotification(CFNotificationCenterGetDarwinNotifyCenter(), WebDatabaseWasDeletedNotification, nullptr, nullptr, true);
}

void WebDatabaseManagerClient::dispatchDidDeleteDatabaseOrigin()
{
    m_isHandlingDeleteDatabaseOriginNotification = true;
    // Send a notification to all apps that an origin has been deleted, so other apps can update their origin maps.
    CFNotificationCenterPostNotification(CFNotificationCenterGetDarwinNotifyCenter(), WebDatabaseOriginWasDeletedNotification, nullptr, nullptr, true);
}

void WebDatabaseManagerClient::newDatabaseOriginWasAdded()
{
    // Locks the WebThread for the rest of the run loop.
    WebThreadLock();

    // If this is the process that added the new origin, its quota map should have been updated
    // and does not need to be invalidated.
    if (m_isHandlingNewDatabaseOriginNotification) {
        m_isHandlingNewDatabaseOriginNotification = false;
        return;
    }
    
    databaseOriginsDidChange();
}

void WebDatabaseManagerClient::databaseWasDeleted()
{
    // Locks the WebThread for the rest of the run loop.
    WebThreadLock();
    
    // If this is the process that added the new origin, its quota map should have been updated
    // and does not need to be invalidated.
    if (m_isHandlingDeleteDatabaseNotification) {
        m_isHandlingDeleteDatabaseNotification = false;
        return;
    }
    
    DatabaseTracker::singleton().removeDeletedOpenedDatabases();
}

void WebDatabaseManagerClient::databaseOriginWasDeleted()
{
    // Locks the WebThread for the rest of the run loop.
    WebThreadLock();
    
    // If this is the process that added the new origin, its quota map should have been updated
    // and does not need to be invalidated.
    if (m_isHandlingDeleteDatabaseOriginNotification) {
        m_isHandlingDeleteDatabaseOriginNotification = false;
        return;
    }

    databaseOriginsDidChange();
}

void WebDatabaseManagerClient::databaseOriginsDidChange()
{
    // Send a notification (within the app) about the origins change.  If an app needs to update its UI when the origins change
    // (such as Safari Settings), it can listen for that notification.
    CFNotificationCenterPostNotification(CFNotificationCenterGetLocalCenter(), WebDatabaseOriginsDidChangeNotification, 0, 0, true);
}

#endif // PLATFORM(IOS_FAMILY)

} // namespace WebKit
