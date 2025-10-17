/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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
#include "IPCConnectionTester.h"

#if ENABLE(IPC_TESTING_API)
#include "IPCConnectionTesterMessages.h"
#include "IPCUtilities.h"

namespace WebKit {

Ref<IPCConnectionTester> IPCConnectionTester::create(IPC::Connection& connection, IPCConnectionTesterIdentifier identifier, IPC::Connection::Handle&& handle)
{
    auto tester = adoptRef(*new IPCConnectionTester(connection, identifier, WTFMove(handle)));
    tester->initialize();
    return tester;
}

IPCConnectionTester::IPCConnectionTester(Ref<IPC::Connection>&& connection, IPCConnectionTesterIdentifier identifier, IPC::Connection::Handle&& handle)
    : m_connection(WTFMove(connection))
    , m_testedConnection(IPC::Connection::createClientConnection(IPC::Connection::Identifier { WTFMove(handle) }))
    , m_identifier(identifier)
{
}

IPCConnectionTester::~IPCConnectionTester() = default;

void IPCConnectionTester::initialize()
{
    m_testedConnection->open(*this);
}

void IPCConnectionTester::stopListeningForIPC(Ref<IPCConnectionTester>&& refFromConnection)
{
    m_testedConnection->invalidate();
}

void IPCConnectionTester::sendAsyncMessages(uint32_t messageCount)
{
    for (uint32_t i = 0; i < messageCount; ++i)
        m_testedConnection->send(Messages::IPCConnectionTester::AsyncMessage(i), 0);
}

void IPCConnectionTester::didClose(IPC::Connection&)
{
}

void IPCConnectionTester::didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t)
{
    ASSERT_NOT_REACHED();
}

void IPCConnectionTester::asyncMessage(uint32_t value)
{
    if (m_previousAsyncMessageValue != value - 1) {
        ASSERT_IS_TESTING_IPC();
        return;
    }
    m_previousAsyncMessageValue = value;
}

void IPCConnectionTester::syncMessage(uint32_t value, CompletionHandler<void(uint32_t)>&& completionHandler)
{
    completionHandler(value + m_previousAsyncMessageValue);
}

}

#endif
