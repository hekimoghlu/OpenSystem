/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

#include "AuthenticatorTransportService.h"
#include "CtapDriver.h"
#include <wtf/RetainPtr.h>
#include <wtf/UniqueRef.h>

namespace WebKit {

class FidoService : public AuthenticatorTransportService, public RefCounted<FidoService> {
    WTF_MAKE_TZONE_ALLOCATED(FidoService);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

protected:
    explicit FidoService(AuthenticatorTransportServiceObserver&);
    void getInfo(Ref<CtapDriver>&&);

private:
    void continueAfterGetInfo(WeakPtr<CtapDriver>&&, Vector<uint8_t>&& info);

    // Keeping drivers alive when they are getting info from devices.
    HashSet<Ref<CtapDriver>> m_drivers;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
