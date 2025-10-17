/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
#import "WebPushToolConnection.h"

#import "DaemonEncoder.h"
#import "DaemonUtilities.h"
#import "PushClientConnectionMessages.h"
#import "WebPushDaemonConnectionConfiguration.h"
#import "WebPushDaemonConstants.h"
#import <WebCore/SecurityOriginData.h>
#import <mach/mach_init.h>
#import <mach/task.h>
#import <pal/spi/cocoa/ServersSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/MainThread.h>
#import <wtf/RetainPtr.h>
#import <wtf/StdLibExtras.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebPushTool {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Connection);

Ref<Connection> Connection::create(PreferTestService preferTestService, String bundleIdentifier, String pushPartition)
{
    return adoptRef(*new Connection(preferTestService, bundleIdentifier, pushPartition));
}

static mach_port_t maybeConnectToService(const char* serviceName)
{
    mach_port_t bsPort;
    task_get_special_port(mach_task_self(), TASK_BOOTSTRAP_PORT, &bsPort);

    mach_port_t servicePort;
    kern_return_t err = bootstrap_look_up(bsPort, serviceName, &servicePort);

    if (err == KERN_SUCCESS)
        return servicePort;

    return MACH_PORT_NULL;
}

Connection::Connection(PreferTestService preferTestService, String bundleIdentifier, String pushPartition)
    : m_bundleIdentifier(bundleIdentifier)
    , m_pushPartition(pushPartition)
{
    if (preferTestService == PreferTestService::Yes)
        m_serviceName = "org.webkit.webpushtestdaemon.service"_s;
    else
        m_serviceName = "com.apple.webkit.webpushd.service"_s;
}

void Connection::connectToService(WaitForServiceToExist waitForServiceToExist)
{

    m_connection = adoptNS(xpc_connection_create_mach_service(m_serviceName, dispatch_get_main_queue(), 0));

    xpc_connection_set_event_handler(m_connection.get(), [](xpc_object_t event) {
        if (event == XPC_ERROR_CONNECTION_INVALID || event == XPC_ERROR_CONNECTION_INTERRUPTED) {
            SAFE_FPRINTF(stderr, "Unexpected XPC connection issue: %s\n", String(event.debugDescription).utf8());
            return;
        }

        RELEASE_ASSERT_NOT_REACHED();
    });

    if (waitForServiceToExist == WaitForServiceToExist::Yes) {
        auto result = maybeConnectToService(m_serviceName);
        if (result == MACH_PORT_NULL)
            SAFE_PRINTF("Waiting for service '%s' to be available\n", m_serviceName);

        while (result == MACH_PORT_NULL) {
            usleep(1000);
            result = maybeConnectToService(m_serviceName);
        }
    }

    SAFE_PRINTF("Connecting to service '%s'\n", m_serviceName);
    xpc_connection_activate(m_connection.get());

    sendAuditToken();
}

void Connection::sendPushMessage(PushMessageForTesting&& message, CompletionHandler<void(String)>&& completionHandler)
{
    printf("Injecting push message\n");

    sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::InjectPushMessageForTesting(WTFMove(message)), WTFMove(completionHandler));
}

void Connection::getPushPermissionState(const String& scope, CompletionHandler<void(WebCore::PushPermissionState)>&& completionHandler)
{
    printf("Getting push permission state\n");

    sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::GetPushPermissionState(WebCore::SecurityOriginData::fromURL(URL { scope })), WTFMove(completionHandler));
}

void Connection::requestPushPermission(const String& scope, CompletionHandler<void(bool)>&& completionHandler)
{
    SAFE_PRINTF("Request push permission state for %s\n", scope.utf8());

    sendWithAsyncReplyWithoutUsingIPCConnection(Messages::PushClientConnection::RequestPushPermission(WebCore::SecurityOriginData::fromURL(URL { scope })), WTFMove(completionHandler));
}

void Connection::sendAuditToken()
{
    audit_token_t token = { 0, 0, 0, 0, 0, 0, 0, 0 };
    mach_msg_type_number_t auditTokenCount = TASK_AUDIT_TOKEN_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_AUDIT_TOKEN, (task_info_t)(&token), &auditTokenCount);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    if (result != KERN_SUCCESS) {
        printf("Unable to get audit token to send\n");
        return;
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    WebKit::WebPushD::WebPushDaemonConnectionConfiguration configuration;
    configuration.bundleIdentifierOverride = m_bundleIdentifier;
    configuration.pushPartitionString = m_pushPartition;

    Vector<uint8_t> tokenVector;
    tokenVector.resize(32);
    memcpySpan(tokenVector.mutableSpan(), asByteSpan(token));
    configuration.hostAppAuditTokenData = WTFMove(tokenVector);

    sendWithoutUsingIPCConnection(Messages::PushClientConnection::InitializeConnection(WTFMove(configuration)));
}

static OSObjectPtr<xpc_object_t> messageDictionaryFromEncoder(UniqueRef<IPC::Encoder>&& encoder)
{
    auto xpcData = WebKit::encoderToXPCData(WTFMove(encoder));
    auto dictionary = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_uint64(dictionary.get(), WebKit::WebPushD::protocolVersionKey, WebKit::WebPushD::protocolVersionValue);
    xpc_dictionary_set_value(dictionary.get(), WebKit::WebPushD::protocolEncodedMessageKey, xpcData.get());

    return dictionary;
}

bool Connection::performSendWithoutUsingIPCConnection(UniqueRef<IPC::Encoder>&& encoder) const
{
    auto dictionary = messageDictionaryFromEncoder(WTFMove(encoder));
    xpc_connection_send_message(m_connection.get(), dictionary.get());

    return true;
}

bool Connection::performSendWithAsyncReplyWithoutUsingIPCConnection(UniqueRef<IPC::Encoder>&& encoder, CompletionHandler<void(IPC::Decoder*)>&& completionHandler) const
{
    auto dictionary = messageDictionaryFromEncoder(WTFMove(encoder));
    xpc_connection_send_message_with_reply(m_connection.get(), dictionary.get(), dispatch_get_main_queue(), makeBlockPtr([completionHandler = WTFMove(completionHandler)] (xpc_object_t reply) mutable {
        if (xpc_get_type(reply) != XPC_TYPE_DICTIONARY) {
            ASSERT_NOT_REACHED();
            return completionHandler(nullptr);
        }
        if (xpc_dictionary_get_uint64(reply, WebKit::WebPushD::protocolVersionKey) != WebKit::WebPushD::protocolVersionValue) {
            ASSERT_NOT_REACHED();
            return completionHandler(nullptr);
        }

        auto data = xpc_dictionary_get_data_span(reply, WebKit::WebPushD::protocolEncodedMessageKey);
        auto decoder = IPC::Decoder::create(data, { });
        ASSERT(decoder);

        completionHandler(decoder.get());
    }).get());

    return true;
}


} // namespace WebPushTool

#endif // ENABLE(WEB_PUSH_NOTIFICATIONS)
