/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#import "WebPushToolMain.h"

#import "PushMessageForTesting.h"
#import "WebPushToolConnection.h"
#import <optional>
#import <wtf/MainThread.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/WTFProcess.h>

#if HAVE(OS_LAUNCHD_JOB) && (PLATFORM(MAC) || PLATFORM(IOS))

using WebKit::WebPushD::PushMessageForTesting;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

__attribute__((__noreturn__))
static void printUsageAndTerminate(NSString *message)
{
    fprintf(stderr, "%s\n\n", message.UTF8String);

    fprintf(stderr, "Usage: webpushtool [options] verb [verb_args]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options is one or more of:\n");
    fprintf(stderr, "  --development\n");
    fprintf(stderr, "    Connects to mach service \"org.webkit.webpushtestdaemon.service\" (Default)\n");
    fprintf(stderr, "  --production\n");
    fprintf(stderr, "    Connects to mach service \"com.apple.webkit.webpushd.service\"\n");
    fprintf(stderr, "  --bundleIdentifier <bundleIdentifier>\n");
    fprintf(stderr, "    Sets connection config to use bundle identifier <bundleIdentifier>.\n");
    fprintf(stderr, "  --pushPartition <partition>\n");
    fprintf(stderr, "    Sets connection config to use push partition <partition>.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "verb is one of:\n");
#if HAVE(OS_LAUNCHD_JOB)
    fprintf(stderr, "  host\n");
    fprintf(stderr, "    Dynamically registers the service with launchd so it is visible to other applications\n");
    fprintf(stderr, "    The service name of the registration depends on either the --development or --production option chosen\n");
#endif
    fprintf(stderr, "  streamDebugMessages\n");
    fprintf(stderr, "    Stream debug messages from webpushd\n");
    fprintf(stderr, "  injectPushMessage <scope URL> <message>\n");
    fprintf(stderr, "    Inject a test push message <message> to the provided --bundleIdentifier and --pushPartition with service worker scope <scope URL>\n");
    fprintf(stderr, "  getPushPermissionState <scope URL>\n");
    fprintf(stderr, "    Gets the permission state for the given service worker scope.\n");
    fprintf(stderr, "  requestPushPermission <scope URL>\n");
    fprintf(stderr, "    Requests permission state for the given service worker scope.\n");
    fprintf(stderr, "\n");

    exitProcess(-1);
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

static std::unique_ptr<PushMessageForTesting> pushMessageFromArguments(NSEnumerator<NSString *> *enumerator)
{
    NSString *registrationString = [enumerator nextObject];
    if (!registrationString)
        return nullptr;

    NSURL *registrationURL = [NSURL URLWithString:registrationString];
    if (!registrationURL)
        return nullptr;

    NSString *message = [enumerator nextObject];
    if (!message)
        return nullptr;

#if ENABLE(DECLARATIVE_WEB_PUSH)
    PushMessageForTesting pushMessage = { { }, { }, registrationURL, message, WebKit::WebPushD::PushMessageDisposition::Legacy, std::nullopt };
#else
    PushMessageForTesting pushMessage = { { }, { }, registrationURL, message, WebKit::WebPushD::PushMessageDisposition::Legacy };
#endif

    return makeUniqueWithoutFastMallocCheck<PushMessageForTesting>(WTFMove(pushMessage));
}

static bool registerDaemonWithLaunchD(WebPushTool::PreferTestService preferTestService)
{
    // For now webpushtool only knows how to host webpushd when they're in the same directory
    // e.g. the build directory of a WebKit contributor.
    NSString *currentExecutablePath = [[NSBundle mainBundle] executablePath];
    NSURL *currentExecutableDirectoryURL = [[NSURL fileURLWithPath:currentExecutablePath isDirectory:NO] URLByDeletingLastPathComponent];
    NSURL *daemonExecutablePathURL = [currentExecutableDirectoryURL URLByAppendingPathComponent:@"webpushd"];

    if (![[NSFileManager defaultManager] fileExistsAtPath:daemonExecutablePathURL.path]) {
        NSLog(@"Daemon executable does not exist at path %@", daemonExecutablePathURL.path);
        return false;
    }

    const char* serviceName = (preferTestService == WebPushTool::PreferTestService::Yes) ? "org.webkit.webpushtestdaemon.service" : "com.apple.webkit.webpushd.service";

    auto plist = adoptNS(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_string(plist.get(), "_ManagedBy", "webpushtool");
    xpc_dictionary_set_string(plist.get(), "Label", "org.webkit.webpushtestdaemon");
    xpc_dictionary_set_bool(plist.get(), "LaunchOnlyOnce", true);
    xpc_dictionary_set_bool(plist.get(), "RootedSimulatorPath", true);

    {
        auto environmentVariables = adoptNS(xpc_dictionary_create(nullptr, nullptr, 0));
        xpc_dictionary_set_string(environmentVariables.get(), "DYLD_FRAMEWORK_PATH", currentExecutableDirectoryURL.fileSystemRepresentation);
        xpc_dictionary_set_string(environmentVariables.get(), "DYLD_LIBRARY_PATH", currentExecutableDirectoryURL.fileSystemRepresentation);
        xpc_dictionary_set_value(plist.get(), "EnvironmentVariables", environmentVariables.get());
    }
    {
        auto machServices = adoptNS(xpc_dictionary_create(nullptr, nullptr, 0));
        xpc_dictionary_set_bool(machServices.get(), serviceName, true);
        xpc_dictionary_set_value(plist.get(), "MachServices", machServices.get());
    }
    {
        auto programArguments = adoptNS(xpc_array_create(nullptr, 0));
#if PLATFORM(MAC)
        xpc_array_set_string(programArguments.get(), XPC_ARRAY_APPEND, daemonExecutablePathURL.fileSystemRepresentation);
#else
        xpc_array_set_string(programArguments.get(), XPC_ARRAY_APPEND, daemonExecutablePathURL.path.fileSystemRepresentation);
#endif
        xpc_array_set_string(programArguments.get(), XPC_ARRAY_APPEND, "--machServiceName");
        xpc_array_set_string(programArguments.get(), XPC_ARRAY_APPEND, serviceName);
        xpc_dictionary_set_value(plist.get(), "ProgramArguments", programArguments.get());
    }

    auto job = adoptNS([[OSLaunchdJob alloc] initWithPlist:plist.get()]);
    NSError *error = nil;
    [job submit:&error];

    if (error) {
        NSLog(@"Error setting up service: %@", error);
        return false;
    }

    return true;
}

namespace WebKit {

class WebPushToolVerb {
public:
    virtual ~WebPushToolVerb() = default;
    virtual void run(WebPushTool::Connection&) = 0;
    virtual void done() { CFRunLoopStop(CFRunLoopGetMain()); }
};

class InjectPushMessageVerb : public WebPushToolVerb {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(InjectPushMessageVerb);
public:
    InjectPushMessageVerb(PushMessageForTesting&& message)
        : m_pushMessage(WTFMove(message)) { }
    ~InjectPushMessageVerb() = default;

    void run(WebPushTool::Connection& connection) override
    {
        auto pushMessage = m_pushMessage;
        pushMessage.targetAppCodeSigningIdentifier = connection.bundleIdentifier();
        pushMessage.pushPartitionString = connection.pushPartition();

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        connection.sendPushMessage(WTFMove(pushMessage), [this, bundleIdentifier = connection.bundleIdentifier(), webClipIdentifier = connection.pushPartition()](String error) mutable {
            if (error.isEmpty())
                printf("Successfully injected push message %s for [bundleID = %s, webClipIdentifier = %s, scope = %s]\n", m_pushMessage.payload.utf8().data(), bundleIdentifier.utf8().data(), webClipIdentifier.utf8().data(), m_pushMessage.registrationURL.string().utf8().data());
            else
                printf("Injected push message with error: %s\n", error.utf8().data());
            done();
        });
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }

private:
    PushMessageForTesting m_pushMessage;
};

class GetPushPermissionStateVerb : public WebPushToolVerb {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(GetPushPermissionStateVerb);
public:
    GetPushPermissionStateVerb(const String& scope)
        : m_scope(scope) { }
    ~GetPushPermissionStateVerb() = default;

    void run(WebPushTool::Connection& connection) override
    {
        connection.getPushPermissionState(m_scope, [this](WebCore::PushPermissionState state) mutable {
            printf("Got push permission status: %u\n", static_cast<unsigned>(state));
            done();
        });
    }

private:
    String m_scope;
};

class RequestPushPermissionVerb : public WebPushToolVerb {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(RequestPushPermissionVerb);
public:
    RequestPushPermissionVerb(const String& scope)
        : m_scope(scope) { }
    ~RequestPushPermissionVerb() = default;

    void run(WebPushTool::Connection& connection) override
    {
        connection.requestPushPermission(m_scope, [this](bool granted) mutable {
            printf("Requested push permission with result: %d\n", granted);
            done();
        });
    }

private:
    String m_scope;
};

int WebPushToolMain(int, char **)
{
    WTF::initializeMainThread();

    auto preferTestService = WebPushTool::PreferTestService::Yes;
    bool host = false;
    std::unique_ptr<WebPushToolVerb> verb;
    RetainPtr<NSString> bundleIdentifier = @"com.apple.WebKit.TestWebKitAPI";
    RetainPtr<NSString> pushPartition;

    @autoreleasepool {
        NSArray *arguments = [[NSProcessInfo processInfo] arguments];
        if (arguments.count == 1)
            printUsageAndTerminate(@"No arguments provided");

        NSEnumerator<NSString *> *enumerator = [[arguments subarrayWithRange:NSMakeRange(1, arguments.count - 1)] objectEnumerator];
        NSString *argument = [enumerator nextObject];
        while (argument) {
            if ([argument isEqualToString:@"--production"])
                preferTestService = WebPushTool::PreferTestService::No;
            else if ([argument isEqualToString:@"--development"])
                preferTestService = WebPushTool::PreferTestService::Yes;
            else if ([argument isEqualToString:@"--bundleIdentifier"])
                bundleIdentifier = [enumerator nextObject];
            else if ([argument isEqualToString:@"--pushPartition"])
                pushPartition = [enumerator nextObject];
            else if ([argument isEqualToString:@"host"])
                host = true;
            else if ([argument isEqualToString:@"streamDebugMessages"])
                execl("/usr/bin/log", "log", "stream", "--debug", "--info", "--process", "webpushd");
            else if ([argument isEqualToString:@"injectPushMessage"]) {
                auto pushMessage = pushMessageFromArguments(enumerator);
                if (!pushMessage)
                    printUsageAndTerminate([NSString stringWithFormat:@"Invalid push arguments specified"]);
                verb = makeUnique<InjectPushMessageVerb>(WTFMove(*pushMessage));
            } else if ([argument isEqualToString:@"getPushPermissionState"]) {
                String scope { [enumerator nextObject] };
                verb = makeUnique<GetPushPermissionStateVerb>(scope);
            } else if ([argument isEqualToString:@"requestPushPermission"]) {
                String scope { [enumerator nextObject] };
                verb = makeUnique<RequestPushPermissionVerb>(scope);
            } else
                printUsageAndTerminate([NSString stringWithFormat:@"Invalid option provided: %@", argument]);

            argument = [enumerator nextObject];
        }
    }

    if (host) {
        if (!registerDaemonWithLaunchD(preferTestService)) {
            NSLog(@"Failed to register webpushd with launchd");
            exit(1);
        }

        NSLog(@"Registered webpushd with launchd");
        exit(0);
    }

    if (!verb)
        printUsageAndTerminate(@"Nothing to do");

    auto connection = WebPushTool::Connection::create(preferTestService, bundleIdentifier.get(), pushPartition.get());
    connection->connectToService(host ? WebPushTool::WaitForServiceToExist::No : WebPushTool::WaitForServiceToExist::Yes);
    verb->run(connection.get());

    CFRunLoopRun();
    return 0;
}

} // namespace WebKit

#else

namespace WebKit {
int WebPushToolMain(int, char **)
{
    return -1;
}
}

#endif // #if HAVE(OS_LAUNCHD_JOB) && (PLATFORM(MAC) || PLATFORM(IOS))
