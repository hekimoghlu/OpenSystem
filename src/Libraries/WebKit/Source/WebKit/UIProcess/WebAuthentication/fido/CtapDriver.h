/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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

#include <WebCore/AuthenticatorTransport.h>
#include <WebCore/FidoConstants.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

class CtapDriver : public RefCountedAndCanMakeWeakPtr<CtapDriver> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(CtapDriver);
    WTF_MAKE_NONCOPYABLE(CtapDriver);
public:
    using ResponseCallback = Function<void(Vector<uint8_t>&&)>;

    virtual ~CtapDriver() = default;

    void setProtocol(fido::ProtocolVersion protocol) { m_protocol = protocol; }

    WebCore::AuthenticatorTransport transport() const { return m_transport; }
    fido::ProtocolVersion protocol() const { return m_protocol; }
    bool isCtap2Protocol() const { return fido::isCtap2Protocol(m_protocol); }
    void setMaxMsgSize(std::optional<uint32_t> maxMsgSize) { m_maxMsgSize = maxMsgSize; }
    bool isValidSize(size_t msgSize) { return !m_maxMsgSize || msgSize <= static_cast<size_t>(*m_maxMsgSize); }

    virtual void transact(Vector<uint8_t>&& data, ResponseCallback&&) = 0;
    virtual void cancel() { };
protected:
    explicit CtapDriver(WebCore::AuthenticatorTransport transport)
        : m_transport(transport)
    { }

private:
    fido::ProtocolVersion m_protocol { fido::ProtocolVersion::kCtap2 };
    WebCore::AuthenticatorTransport m_transport;
    std::optional<uint32_t> m_maxMsgSize;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
