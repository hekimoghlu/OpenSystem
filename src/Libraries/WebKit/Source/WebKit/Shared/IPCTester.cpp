/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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
#include "config.h"
#include "IPCTester.h"

#if ENABLE(IPC_TESTING_API)
#include "Connection.h"
#include "Decoder.h"
#include "IPCConnectionTester.h"
#include "IPCStreamTester.h"
#include "IPCTesterMessages.h"
#include "IPCTesterReceiverMessages.h"
#include "IPCUtilities.h"

#include <WebCore/ExceptionData.h>
#include <atomic>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/threads/BinarySemaphore.h>

// The tester API.
extern "C" {
// Returns 0 if driver should continue.
typedef int (*WKMessageTestSendMessageFunc)(std::span<const uint8_t> buffer, void* context);
typedef void (*WKMessageTestDriverFunc)(WKMessageTestSendMessageFunc sendMessageFunc, void* context);
}

namespace {
struct SendMessageContext {
    Ref<IPC::Connection> connection;
    std::atomic<bool>& shouldStop;
};

}

extern "C" {

static void defaultTestDriver(WKMessageTestSendMessageFunc sendMessageFunc, void* context)
{
    Vector<uint8_t> data(1000);
    for (unsigned i = 0; i < 1000; ++i) {
        cryptographicallyRandomValues(data.mutableSpan());
        if (sendMessageFunc(data.span(), context))
            return;
    }
}

static int sendTestMessage(std::span<const uint8_t> buffer, void* context)
{
    auto messageContext = reinterpret_cast<SendMessageContext*>(context);
    if (messageContext->shouldStop)
        return 1;
    Ref testedConnection = messageContext->connection;
    if (!testedConnection->isValid())
        return 1;
    BinarySemaphore semaphore;
    auto decoder = IPC::Decoder::create(buffer, [&semaphore] (std::span<const uint8_t>) { semaphore.signal(); }, { }); // NOLINT
    if (decoder) {
        testedConnection->dispatchIncomingMessageForTesting(makeUniqueRefFromNonNullUniquePtr(WTFMove(decoder)));
        semaphore.wait();
    }
    return 0;
}

}

namespace WebKit {

static WKMessageTestDriverFunc messageTestDriver(String&& driverName)
{
    if (driverName.isEmpty() || driverName == "default"_s)
        driverName = String::fromUTF8(getenv("WEBKIT_MESSAGE_TEST_DEFAULT_DRIVER"));
    if (driverName.isEmpty() || driverName == "default"_s)
        return defaultTestDriver;
    auto testDriver = reinterpret_cast<WKMessageTestDriverFunc>(dlsym(RTLD_DEFAULT, driverName.utf8().data()));
    RELEASE_ASSERT(testDriver);
    return testDriver;
}

static void runMessageTesting(IPC::Connection& connection, std::atomic<bool>& shouldStop, String&& driverName)
{
    connection.setIgnoreInvalidMessageForTesting();
    SendMessageContext context { connection, shouldStop };
    auto driver = messageTestDriver(WTFMove(driverName));
    driver(sendTestMessage, &context);
}

Ref<IPCTester> IPCTester::create()
{
    return adoptRef(*new IPCTester);
}

IPCTester::IPCTester() = default;

IPCTester::~IPCTester()
{
    stopIfNeeded();
}

void IPCTester::startMessageTesting(IPC::Connection& connection, String&& driverName)
{
    if (!m_testQueue)
        m_testQueue = WorkQueue::create("IPC testing work queue"_s);
    m_testQueue->dispatch([connection = Ref { connection }, &shouldStop = m_shouldStop, driverName = WTFMove(driverName)]() mutable {
        IPC::startTestingIPC();
        runMessageTesting(connection, shouldStop, WTFMove(driverName));
        IPC::stopTestingIPC();
    });
}

void IPCTester::stopMessageTesting(CompletionHandler<void()>&& completionHandler)
{
    stopIfNeeded();
    completionHandler();
}

void IPCTester::createStreamTester(IPC::Connection& connection, IPCStreamTesterIdentifier identifier, IPC::StreamServerConnection::Handle&& serverConnection)
{
    auto addResult = m_streamTesters.ensure(identifier, [&] {
        return IPC::ScopedActiveMessageReceiveQueue<IPCStreamTester> { IPCStreamTester::create(identifier, WTFMove(serverConnection), connection.ignoreInvalidMessageForTesting()) };
    });
    ASSERT_UNUSED(addResult, addResult.isNewEntry || IPC::isTestingIPC());
}

void IPCTester::releaseStreamTester(IPCStreamTesterIdentifier identifier, CompletionHandler<void()>&& completionHandler)
{
    m_streamTesters.remove(identifier);
    completionHandler();
}

void IPCTester::sendSameSemaphoreBack(IPC::Connection& connection, IPC::Semaphore&& semaphore)
{
    connection.send(Messages::IPCTester::SendSameSemaphoreBack(semaphore), 0);
}

void IPCTester::sendSemaphoreBackAndSignalProtocol(IPC::Connection& connection, IPC::Semaphore&& semaphore)
{
    IPC::Semaphore newSemaphore;
    connection.send(Messages::IPCTester::SendSemaphoreBackAndSignalProtocol(newSemaphore), 0);
    if (!semaphore.waitFor(10_s)) {
        ASSERT_IS_TESTING_IPC();
        return;
    }
    newSemaphore.signal();
    // Wait for protocol commit. Otherwise newSemaphore will be destroyed, and the waiter on the other side
    // will fail to wait.
    if (!semaphore.waitFor(10_s)) {
        ASSERT_IS_TESTING_IPC();
        return;
    }
}

void IPCTester::sendAsyncMessageToReceiver(IPC::Connection& connection, uint32_t arg0)
{
    connection.sendWithAsyncReply(Messages::IPCTesterReceiver::AsyncMessage(arg0 + 1), [arg0](uint32_t newArg0) {
        ASSERT_UNUSED(arg0, newArg0 == arg0 + 2);
    }, 0);
}

void IPCTester::createConnectionTester(IPC::Connection& connection, IPCConnectionTesterIdentifier identifier, IPC::Connection::Handle&& testedConnectionIdentifier)
{
    auto addResult = m_connectionTesters.ensure(identifier, [&] {
        return IPC::ScopedActiveMessageReceiveQueue<IPCConnectionTester> { IPCConnectionTester::create(connection, identifier, WTFMove(testedConnectionIdentifier)) };
    });
    ASSERT_UNUSED(addResult, addResult.isNewEntry || IPC::isTestingIPC());
}

void IPCTester::createConnectionTesterAndSendAsyncMessages(IPC::Connection& connection, IPCConnectionTesterIdentifier identifier, IPC::Connection::Handle&& testedConnectionIdentifier, uint32_t messageCount)
{
    auto addResult = m_connectionTesters.ensure(identifier, [&] {
        return IPC::ScopedActiveMessageReceiveQueue<IPCConnectionTester> { IPCConnectionTester::create(connection, identifier, WTFMove(testedConnectionIdentifier)) };
    });
    if (!addResult.isNewEntry) {
        ASSERT_IS_TESTING_IPC();
        return;
    }
    addResult.iterator->value->sendAsyncMessages(messageCount);
}

void IPCTester::releaseConnectionTester(IPCConnectionTesterIdentifier identifier, CompletionHandler<void()>&& completionHandler)
{
    m_connectionTesters.remove(identifier);
    completionHandler();
}

void IPCTester::asyncPing(uint32_t value, CompletionHandler<void(uint32_t)>&& completionHandler)
{
    completionHandler(value + 1);
}

void IPCTester::syncPing(IPC::Connection&, uint32_t value, CompletionHandler<void(uint32_t)>&& completionHandler)
{
    completionHandler(value + 1);
}

void IPCTester::syncPingEmptyReply(IPC::Connection&, uint32_t value, CompletionHandler<void()>&& completionHandler)
{
    UNUSED_PARAM(value);
    completionHandler();
}

void IPCTester::asyncOptionalExceptionData(IPC::Connection&, bool sendEngaged, CompletionHandler<void(std::optional<WebCore::ExceptionData>, String)>&& completionHandler)
{
    if (sendEngaged) {
        completionHandler(WebCore::ExceptionData { WebCore::ExceptionCode::WrongDocumentError, "m"_s }, "a"_s);
        return;
    }
    completionHandler(std::nullopt, "b"_s);
}

void IPCTester::stopIfNeeded()
{
    if (RefPtr testQueue = m_testQueue) {
        m_shouldStop = true;
        testQueue->dispatchSync([] { });
        m_testQueue = nullptr;
    }
}

} // namespace WebKit

#endif
