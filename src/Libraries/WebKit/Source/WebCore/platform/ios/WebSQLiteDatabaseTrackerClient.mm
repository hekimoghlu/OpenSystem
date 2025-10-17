/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#import "WebSQLiteDatabaseTrackerClient.h"

#if PLATFORM(IOS_FAMILY)

#import "WebBackgroundTaskController.h"
#import <WebCore/DatabaseTracker.h>
#import <WebCore/SQLiteDatabaseTracker.h>
#import <wtf/Lock.h>
#import <wtf/MainThread.h>
#import <wtf/NeverDestroyed.h>

@interface WebDatabaseTransactionBackgroundTaskController : NSObject
+ (void)startBackgroundTask;
+ (void)endBackgroundTask;
@end

namespace WebCore {

const Seconds hysteresisDuration { 2_s };

WebSQLiteDatabaseTrackerClient& WebSQLiteDatabaseTrackerClient::sharedWebSQLiteDatabaseTrackerClient()
{
    static NeverDestroyed<WebSQLiteDatabaseTrackerClient> client;
    return client;
}

WebSQLiteDatabaseTrackerClient::WebSQLiteDatabaseTrackerClient()
    : m_hysteresis([this](PAL::HysteresisState state) { hysteresisUpdated(state); }, hysteresisDuration)
{
    ASSERT(pthread_main_np());
}

WebSQLiteDatabaseTrackerClient::~WebSQLiteDatabaseTrackerClient()
{
}

void WebSQLiteDatabaseTrackerClient::willBeginFirstTransaction()
{
    RunLoop::main().dispatch([this] {
        m_hysteresis.start();
    });
}

void WebSQLiteDatabaseTrackerClient::didFinishLastTransaction()
{
    RunLoop::main().dispatch([this] {
        m_hysteresis.stop();
    });
}

void WebSQLiteDatabaseTrackerClient::hysteresisUpdated(PAL::HysteresisState state)
{
    ASSERT(pthread_main_np());
    if (state == PAL::HysteresisState::Started)
        [WebDatabaseTransactionBackgroundTaskController startBackgroundTask];
    else
        [WebDatabaseTransactionBackgroundTaskController endBackgroundTask];
}

}

static Lock transactionBackgroundTaskIdentifierLock;

static NSUInteger transactionBackgroundTaskIdentifier WTF_GUARDED_BY_LOCK(transactionBackgroundTaskIdentifierLock);

static void setTransactionBackgroundTaskIdentifier(NSUInteger identifier) WTF_REQUIRES_LOCK(transactionBackgroundTaskIdentifierLock)
{
    transactionBackgroundTaskIdentifier = identifier;
}

static NSUInteger getTransactionBackgroundTaskIdentifier() WTF_REQUIRES_LOCK(transactionBackgroundTaskIdentifierLock)
{
    static dispatch_once_t pred;
    dispatch_once(&pred, ^ {
        setTransactionBackgroundTaskIdentifier([[WebBackgroundTaskController sharedController] invalidBackgroundTaskIdentifier]);
    });

    return transactionBackgroundTaskIdentifier;
}

@implementation WebDatabaseTransactionBackgroundTaskController

+ (void)startBackgroundTask
{
    Locker locker { transactionBackgroundTaskIdentifierLock };

    // If there's already an existing background task going on, there's no need to start a new one.
    WebBackgroundTaskController *backgroundTaskController = [WebBackgroundTaskController sharedController];
    if (getTransactionBackgroundTaskIdentifier() != [backgroundTaskController invalidBackgroundTaskIdentifier])
        return;

    setTransactionBackgroundTaskIdentifier([backgroundTaskController startBackgroundTaskWithExpirationHandler:(^ {
        WebCore::DatabaseTracker::singleton().closeAllDatabases(WebCore::CurrentQueryBehavior::Interrupt);
        [self endBackgroundTask];
    })]);
}

+ (void)endBackgroundTask
{
    Locker locker { transactionBackgroundTaskIdentifierLock };

    // It is possible that we were unable to start the background task when the first transaction began.
    // Don't try to end the task in that case.
    // It is also possible we finally finish the last transaction right when the background task expires
    // and this will end up being called twice for the same background task. transactionBackgroundTaskIdentifier
    // will be invalid for the second caller.
    WebBackgroundTaskController *backgroundTaskController = [WebBackgroundTaskController sharedController];
    if (getTransactionBackgroundTaskIdentifier() == [backgroundTaskController invalidBackgroundTaskIdentifier])
        return;

    [backgroundTaskController endBackgroundTaskWithIdentifier:getTransactionBackgroundTaskIdentifier()];
    setTransactionBackgroundTaskIdentifier([backgroundTaskController invalidBackgroundTaskIdentifier]);
}

@end

#endif // PLATFORM(IOS_FAMILY)
