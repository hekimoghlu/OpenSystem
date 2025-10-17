/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#if ENABLE(WEB_PUSH_NOTIFICATIONS)

#import "WebPushDaemonMain.h"

#import "AuxiliaryProcess.h"
#import "DaemonConnection.h"
#import "DaemonDecoder.h"
#import "DaemonEncoder.h"
#import "DaemonUtilities.h"
#import "LogInitialization.h"
#import "WebPushDaemon.h"
#import <Foundation/Foundation.h>
#import <WebCore/LogInitialization.h>
#import <getopt.h>
#import <pal/spi/cf/CFUtilitiesSPI.h>
#import <pal/spi/cocoa/CoreServicesSPI.h>
#import <wtf/LogInitialization.h>
#import <wtf/MainThread.h>
#import <wtf/OSObjectPtr.h>
#import <wtf/WTFProcess.h>
#import <wtf/spi/darwin/XPCSPI.h>
#import <wtf/text/MakeString.h>

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/WebPushDaemonMainAdditions.mm>)
#import <WebKitAdditions/WebPushDaemonMainAdditions.mm>
#endif

#if !defined(WEB_PUSH_DAEMON_MAIN_ADDITIONS)
#define WEB_PUSH_DAEMON_MAIN_ADDITIONS
#endif

using WebKit::Daemon::EncodedMessage;
using WebPushD::WebPushDaemon;

static const ASCIILiteral entitlementName = "com.apple.private.webkit.webpush"_s;

#if ENABLE(RELOCATABLE_WEBPUSHD)
static const ASCIILiteral defaultMachServiceName = "com.apple.webkit.webpushd.relocatable.service"_s;
static const ASCIILiteral defaultIncomingPushServiceName = "com.apple.aps.webkit.webpushd.relocatable.incoming-push"_s;
#else
static const ASCIILiteral defaultMachServiceName = "com.apple.webkit.webpushd.service"_s;
static const ASCIILiteral defaultIncomingPushServiceName = "com.apple.aps.webkit.webpushd.incoming-push"_s;
#endif

namespace WebPushD {

static void connectionEventHandler(xpc_object_t request)
{
    WebPushDaemon::singleton().connectionEventHandler(request);
}

static void connectionAdded(xpc_connection_t connection)
{
    WebPushDaemon::singleton().connectionAdded(connection);
}

static void connectionRemoved(xpc_connection_t connection)
{
    WebPushDaemon::singleton().connectionRemoved(connection);
}

} // namespace WebPushD

using WebPushD::connectionEventHandler;
using WebPushD::connectionAdded;
using WebPushD::connectionRemoved;

namespace WebKit {

static void applySandbox()
{
#if PLATFORM(MAC)
#if ENABLE(RELOCATABLE_WEBPUSHD)
    static ASCIILiteral profileName = "/com.apple.WebKit.webpushd.relocatable.mac.sb"_s;
    static ASCIILiteral userDirectorySuffix = "com.apple.webkit.webpushd.relocatable"_s;
#else
    static ASCIILiteral profileName = "/com.apple.WebKit.webpushd.mac.sb"_s;
    static ASCIILiteral userDirectorySuffix = "com.apple.webkit.webpushd"_s;
#endif
    NSBundle *bundle = [NSBundle bundleForClass:NSClassFromString(@"WKWebView")];
    auto profilePath = makeString(String([bundle resourcePath]), profileName);
    if (FileSystem::fileExists(profilePath)) {
        AuxiliaryProcess::applySandboxProfileForDaemon(profilePath, userDirectorySuffix);
        return;
    }

    auto oldProfilePath = makeString(String([bundle resourcePath]), "/com.apple.WebKit.webpushd.sb"_s);
    AuxiliaryProcess::applySandboxProfileForDaemon(oldProfilePath, "com.apple.webkit.webpushd"_s);
#endif
}

int WebPushDaemonMain(int argc, char** argv)
{
    @autoreleasepool {
        WTF::initializeMainThread();

        auto transaction = adoptOSObject(os_transaction_create("com.apple.webkit.webpushd.push-service-main"));
        auto peerEntitlementName = entitlementName;

#if ENABLE(CFPREFS_DIRECT_MODE)
        _CFPrefsSetDirectModeEnabled(YES);
#endif
        applySandbox();

#if PLATFORM(IOS) && !PLATFORM(IOS_SIMULATOR)
        if (!_set_user_dir_suffix("com.apple.webkit.webpushd")) {
            auto error = errno;
            auto errorMessage = strerror(error);
            os_log_error(OS_LOG_DEFAULT, "Failed to set temp dir: %{public}s (%d)", errorMessage, error);
            exit(1);
        }
        (void)NSTemporaryDirectory();
#endif

#if !LOG_DISABLED || !RELEASE_LOG_DISABLED
        WTF::logChannels().initializeLogChannelsIfNecessary();
        WebCore::logChannels().initializeLogChannelsIfNecessary();
        WebKit::logChannels().initializeLogChannelsIfNecessary();
#endif // !LOG_DISABLED || !RELEASE_LOG_DISABLED

        static struct option options[] = {
            { "machServiceName", required_argument, 0, 'm' },
            { "incomingPushServiceName", required_argument, 0, 'p' },
            { "useMockPushService", no_argument, 0, 'f' }
        };

        const char* machServiceName = defaultMachServiceName;
        const char* incomingPushServiceName = defaultIncomingPushServiceName;
        bool useMockPushService = false;

        int c;
        int optionIndex;
        while ((c = getopt_long(argc, argv, "", options, &optionIndex)) != -1) {
            switch (c) {
            case 'm':
                machServiceName = optarg;
                break;
            case 'p':
                incomingPushServiceName = optarg;
                break;
            case 'f':
                useMockPushService = true;
                break;
            default:
                fprintf(stderr, "Unknown option: %c\n", optopt);
                exitProcess(1);
            }
        }

        WEB_PUSH_DAEMON_MAIN_ADDITIONS;

        WebKit::startListeningForMachServiceConnections(machServiceName, peerEntitlementName, connectionAdded, connectionRemoved, connectionEventHandler);

        if (useMockPushService)
            ::WebPushD::WebPushDaemon::singleton().startMockPushService();
        else {
            String libraryPath = NSSearchPathForDirectoriesInDomains(NSLibraryDirectory, NSUserDomainMask, YES)[0];

#if ENABLE(RELOCATABLE_WEBPUSHD)
            String pushDatabasePath = FileSystem::pathByAppendingComponents(libraryPath, { "WebKit"_s, "WebPush"_s, "PushDatabase.relocatable.db"_s });
#else
            String pushDatabasePath = FileSystem::pathByAppendingComponents(libraryPath, { "WebKit"_s, "WebPush"_s, "PushDatabase.db"_s });
#endif

            String webClipCachePath = FileSystem::pathByAppendingComponents(libraryPath, { "WebKit"_s, "WebPush"_s, "WebClipCache.plist"_s });

            ::WebPushD::WebPushDaemon::singleton().startPushService(String::fromLatin1(incomingPushServiceName), pushDatabasePath, webClipCachePath);
        }
    }
    CFRunLoopRun();
    return 0;
}

} // namespace WebKit

#endif // ENABLE(WEB_PUSH_NOTIFICATIONS)

