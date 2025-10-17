/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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

#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>

OBJC_CLASS TKSmartCard;

namespace WebKit {

class CcidService;
using DataReceivedCallback = Function<void(Vector<uint8_t>&&)>;

class CcidConnection : public RefCountedAndCanMakeWeakPtr<CcidConnection> {
public:
    static Ref<CcidConnection> create(RetainPtr<TKSmartCard>&&, CcidService&);
    ~CcidConnection();

    void transact(Vector<uint8_t>&& data, DataReceivedCallback&&) const;
    void stop() const;
    bool contactless() const { return m_contactless; };

private:
    CcidConnection(RetainPtr<TKSmartCard>&&, CcidService&);

    void restartPolling();
    void startPolling();

    void detectContactless();

    void trySelectFidoApplet();

    RetainPtr<TKSmartCard> m_smartCard;
    WeakPtr<CcidService> m_service;
    RunLoop::Timer m_retryTimer;
    bool m_contactless { false };
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
