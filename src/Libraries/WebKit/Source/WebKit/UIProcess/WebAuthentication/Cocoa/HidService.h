/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#include <pal/spi/cocoa/IOKitSPI.h>
#include <wtf/RetainPtr.h>

namespace WebKit {

class HidConnection;

class HidService : public FidoService {
public:
    static Ref<HidService> create(AuthenticatorTransportServiceObserver&);
    ~HidService();

    void deviceAdded(IOHIDDeviceRef);

protected:
    explicit HidService(AuthenticatorTransportServiceObserver&);

private:
    void startDiscoveryInternal() final;

    // Overrided by MockHidService.
    virtual void platformStartDiscovery();
    virtual Ref<HidConnection> createHidConnection(IOHIDDeviceRef) const;

    RetainPtr<IOHIDManagerRef> m_manager;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
