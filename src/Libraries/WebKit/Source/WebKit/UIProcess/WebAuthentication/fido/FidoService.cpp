/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#include "FidoService.h"

#if ENABLE(WEB_AUTHN)

#include "CtapAuthenticator.h"
#include "CtapDriver.h"
#include "Logging.h"
#include "U2fAuthenticator.h"
#include <WebCore/DeviceRequestConverter.h>
#include <WebCore/DeviceResponseConverter.h>
#include <WebCore/FidoConstants.h>
#include <WebCore/FidoHidMessage.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/Base64.h>

#define CTAP_RELEASE_LOG(fmt, ...) RELEASE_LOG(WebAuthn, "%p - FidoService::" fmt, this, ##__VA_ARGS__)

namespace WebKit {
using namespace fido;

WTF_MAKE_TZONE_ALLOCATED_IMPL(FidoService);

FidoService::FidoService(AuthenticatorTransportServiceObserver& observer)
    : AuthenticatorTransportService(observer)
{
}

void FidoService::getInfo(Ref<CtapDriver>&& driver)
{
    // Get authenticator info from the device.
    driver->transact(encodeEmptyAuthenticatorRequest(CtapRequestCommand::kAuthenticatorGetInfo), [weakThis = WeakPtr { *this }, weakDriver = WeakPtr { driver.get() }] (Vector<uint8_t>&& response) mutable {
        ASSERT(RunLoop::isMain());
        if (!weakThis)
            return;
        weakThis->continueAfterGetInfo(WTFMove(weakDriver), WTFMove(response));
    });
    auto addResult = m_drivers.add(WTFMove(driver));
    ASSERT_UNUSED(addResult, addResult.isNewEntry);
}

void FidoService::continueAfterGetInfo(WeakPtr<CtapDriver>&& weakDriver, Vector<uint8_t>&& response)
{
    if (!weakDriver)
        return;

    RefPtr driver = m_drivers.take(weakDriver.get());
    if (!driver || !observer() || response.isEmpty())
        return;

    CTAP_RELEASE_LOG("Got response from getInfo: %s", base64EncodeToString(response).utf8().data());

    auto info = readCTAPGetInfoResponse(response);
    if (info && info->versions().find(ProtocolVersion::kCtap2) != info->versions().end()) {
        driver->setMaxMsgSize(info->maxMsgSize());
        observer()->authenticatorAdded(CtapAuthenticator::create(driver.releaseNonNull(), WTFMove(*info)));
        return;
    }
    driver->setProtocol(ProtocolVersion::kU2f);
    observer()->authenticatorAdded(U2fAuthenticator::create(driver.releaseNonNull()));
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
