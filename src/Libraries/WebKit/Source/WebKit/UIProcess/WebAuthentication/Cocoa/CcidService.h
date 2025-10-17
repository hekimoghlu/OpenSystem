/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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

#include "FidoService.h"
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>

OBJC_CLASS NSArray;
OBJC_CLASS TKSmartCardSlot;
OBJC_CLASS TKSmartCard;
OBJC_CLASS _WKSmartCardSlotObserver;
OBJC_CLASS _WKSmartCardSlotStateObserver;

namespace WebKit {

class CcidConnection;

class CcidService : public FidoService {
public:
    static Ref<CcidService> create(AuthenticatorTransportServiceObserver&);
    ~CcidService();

    static bool isAvailable();

    void didConnectTag();

    void updateSlots(NSArray *slots);
    void onValidCard(RetainPtr<TKSmartCard>&&);

protected:
    explicit CcidService(AuthenticatorTransportServiceObserver&);

private:
    void startDiscoveryInternal() final;
    void restartDiscoveryInternal() final;

    void removeObservers();

    virtual void platformStartDiscovery();

    RunLoop::Timer m_restartTimer;
    RefPtr<CcidConnection> m_connection;
    RetainPtr<_WKSmartCardSlotObserver> m_slotsObserver;
    HashMap<String, RetainPtr<_WKSmartCardSlotStateObserver>> m_slotObservers;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
