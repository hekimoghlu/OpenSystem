/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#pragma once

#if ENABLE(WEB_AUTHN)

#include "CtapDriver.h"
#include "HidConnection.h"
#include <WebCore/FidoHidMessage.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebKit {

class CtapHidDriver;

// Worker is the helper that maintains the transaction.
// https://fidoalliance.org/specs/fido-v2.0-ps-20170927/fido-client-to-authenticator-protocol-v2.0-ps-20170927.html#arbitration
// FSM: Idle => Write => Read.
class CtapHidDriverWorker : public CanMakeWeakPtr<CtapHidDriverWorker> {
    WTF_MAKE_TZONE_ALLOCATED(CtapHidDriverWorker);
    WTF_MAKE_NONCOPYABLE(CtapHidDriverWorker);
public:
    using MessageCallback = Function<void(std::optional<fido::FidoHidMessage>&&)>;

    enum class State : uint8_t  {
        Idle,
        Write,
        Read
    };

    CtapHidDriverWorker(CtapHidDriver&, Ref<HidConnection>&&);
    ~CtapHidDriverWorker();

    void transact(fido::FidoHidMessage&&, MessageCallback&&);
    void cancel(fido::FidoHidMessage&&);

    void ref() const;
    void deref() const;

private:
    void write(HidConnection::DataSent);
    void read(const Vector<uint8_t>&);
    void returnMessage();
    void reset();

    Ref<HidConnection> protectedConnection() { return m_connection; }

    WeakRef<CtapHidDriver> m_driver;
    Ref<HidConnection> m_connection;
    State m_state { State::Idle };
    std::optional<fido::FidoHidMessage> m_requestMessage;
    std::optional<fido::FidoHidMessage> m_responseMessage;
    MessageCallback m_callback;
};

// The following implements the CTAP HID protocol:
// https://fidoalliance.org/specs/fido-v2.0-ps-20170927/fido-client-to-authenticator-protocol-v2.0-ps-20170927.html#usb
// FSM: Idle => AllocateChannel => Ready
class CtapHidDriver final : public CtapDriver {
public:
    enum class State : uint8_t {
        Idle,
        AllocateChannel,
        Ready,
        // FIXME(191528)
        Busy
    };

    static Ref<CtapHidDriver> create(Ref<HidConnection>&&);

    void transact(Vector<uint8_t>&& data, ResponseCallback&&) final;
    void cancel() final;

private:
    explicit CtapHidDriver(Ref<HidConnection>&&);

    void continueAfterChannelAllocated(std::optional<fido::FidoHidMessage>&&);
    void continueAfterResponseReceived(std::optional<fido::FidoHidMessage>&&);
    void returnResponse(Vector<uint8_t>&&);
    void reset();

    Ref<CtapHidDriverWorker> protectedWorker() const { return m_worker.get(); }

    const UniqueRef<CtapHidDriverWorker> m_worker;
    State m_state { State::Idle };
    uint32_t m_channelId { fido::kHidBroadcastChannel };
    // One request at a time.
    Vector<uint8_t> m_requestData;
    ResponseCallback m_responseCallback;
    Vector<uint8_t> m_nonce;
};

inline void CtapHidDriverWorker::ref() const
{
    m_driver->ref();
}

inline void CtapHidDriverWorker::deref() const
{
    m_driver->deref();
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
