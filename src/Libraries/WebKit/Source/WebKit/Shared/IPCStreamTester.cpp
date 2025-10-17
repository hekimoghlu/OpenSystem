/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#include "IPCStreamTester.h"

#if ENABLE(IPC_TESTING_API)

#include "Decoder.h"
#include "IPCStreamTesterMessages.h"
#include "IPCStreamTesterProxyMessages.h"
#include "IPCUtilities.h"
#include "StreamConnectionWorkQueue.h"
#include "StreamServerConnection.h"
#include <wtf/WTFProcess.h>

#if USE(FOUNDATION)
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace WebKit {

RefPtr<IPCStreamTester> IPCStreamTester::create(IPCStreamTesterIdentifier identifier, IPC::StreamServerConnection::Handle&& connectionHandle, bool ignoreInvalidMessageForTesting)
{
    auto tester = adoptRef(*new IPCStreamTester(identifier, WTFMove(connectionHandle), ignoreInvalidMessageForTesting));
    tester->initialize();
    return tester;
}

IPCStreamTester::IPCStreamTester(IPCStreamTesterIdentifier identifier, IPC::StreamServerConnection::Handle&& connectionHandle, bool ignoreInvalidMessageForTesting)
    : m_workQueue(IPC::StreamConnectionWorkQueue::create("IPCStreamTester work queue"_s))
    , m_streamConnection(IPC::StreamServerConnection::tryCreate(WTFMove(connectionHandle), { ignoreInvalidMessageForTesting }).releaseNonNull())
    , m_identifier(identifier)
{
}

IPCStreamTester::~IPCStreamTester() = default;

void IPCStreamTester::initialize()
{
    protectedWorkQueue()->dispatch([this] {
        m_streamConnection->open(protectedWorkQueue());
        m_streamConnection->startReceivingMessages(*this, Messages::IPCStreamTester::messageReceiverName(), m_identifier.toUInt64());
        m_streamConnection->send(Messages::IPCStreamTesterProxy::WasCreated(workQueue().wakeUpSemaphore(), m_streamConnection->clientWaitSemaphore()), m_identifier);
    });
}

void IPCStreamTester::stopListeningForIPC(Ref<IPCStreamTester>&& refFromConnection)
{
    Ref workQueue = m_workQueue;
    workQueue->dispatch([this] {
        m_streamConnection->stopReceivingMessages(Messages::IPCStreamTester::messageReceiverName(), m_identifier.toUInt64());
        m_streamConnection->invalidate();
    });
    workQueue->stopAndWaitForCompletion();
}

void IPCStreamTester::syncMessageReturningSharedMemory1(uint32_t byteCount, CompletionHandler<void(std::optional<WebCore::SharedMemory::Handle>&&)>&& completionHandler)
{
    auto result = [&]() -> std::optional<WebCore::SharedMemory::Handle> {
        auto sharedMemory = WebCore::SharedMemory::allocate(byteCount);
        if (!sharedMemory)
            return std::nullopt;
        auto handle = sharedMemory->createHandle(WebCore::SharedMemory::Protection::ReadOnly);
        if (!handle)
            return std::nullopt;
        auto data = sharedMemory->mutableSpan();
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = i;
        return WTFMove(*handle);
    }();
    completionHandler(WTFMove(result));
}

void IPCStreamTester::syncMessageEmptyReply(uint32_t, CompletionHandler<void()>&& completionHandler)
{
    completionHandler();
}

void IPCStreamTester::syncMessage(uint32_t value, CompletionHandler<void(uint32_t)>&& completionHandler)
{
    completionHandler(value);
}

void IPCStreamTester::syncMessageNotStreamEncodableReply(uint32_t value, CompletionHandler<void(uint32_t)>&& completionHandler)
{
    completionHandler(value);
}

void IPCStreamTester::syncMessageNotStreamEncodableBoth(uint32_t value, CompletionHandler<void(uint32_t)>&& completionHandler)
{
    completionHandler(value);
}

void IPCStreamTester::syncCrashOnZero(int32_t value, CompletionHandler<void(int32_t)>&& completionHandler)
{
    if (!value) {
        // Use exit so that we don't leave a crash report.
        terminateProcess(EXIT_SUCCESS);
    }
    completionHandler(value);
}

void IPCStreamTester::asyncPing(uint32_t value, CompletionHandler<void(uint32_t)>&& completionHandler)
{
    completionHandler(value + 1);
}

void IPCStreamTester::emptyMessage()
{
}

#if USE(FOUNDATION)

namespace {
struct UseCountHolder {
    std::shared_ptr<bool> value;
};
}

static void releaseUseCountHolder(CFAllocatorRef, const void* value)
{
    delete static_cast<const UseCountHolder*>(value);
}

#endif

void IPCStreamTester::checkAutoreleasePool(CompletionHandler<void(int32_t)>&& completionHandler)
{
    if (!m_autoreleasePoolCheckValue)
        m_autoreleasePoolCheckValue = std::make_shared<bool>(true);
    completionHandler(m_autoreleasePoolCheckValue.use_count());

#if USE(FOUNDATION)
    static const CFArrayCallBacks arrayCallbacks {
        .version = 0,
        .retain = nullptr,
        .release = releaseUseCountHolder,
        .copyDescription = nullptr,
        .equal = nullptr,
    };
    const void* values[] = { new UseCountHolder { m_autoreleasePoolCheckValue } };
    CFArrayRef releaseDetector = CFArrayCreate(kCFAllocatorDefault, values, 1, &arrayCallbacks);
    CFAutorelease(releaseDetector);
#endif
}
}

#endif
