/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
#import "config.h"
#import "CcidConnection.h"

#if ENABLE(WEB_AUTHN)
#import "CcidService.h"
#import <CryptoTokenKit/TKSmartCard.h>
#import <WebCore/FidoConstants.h>
#import <wtf/Algorithms.h>
#import <wtf/BlockPtr.h>
#import <wtf/StdLibExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {
using namespace fido;

Ref<CcidConnection> CcidConnection::create(RetainPtr<TKSmartCard>&& smartCard, CcidService& service)
{
    return adoptRef(*new CcidConnection(WTFMove(smartCard), service));
}

CcidConnection::CcidConnection(RetainPtr<TKSmartCard>&& smartCard, CcidService& service)
    : m_smartCard(WTFMove(smartCard))
    , m_service(service)
    , m_retryTimer(RunLoop::main(), this, &CcidConnection::startPolling)
{
    startPolling();
}

CcidConnection::~CcidConnection()
{
    stop();
}

const uint8_t kGetUidCommand[] = {
    0xFF, 0xCA, 0x00, 0x00, 0x00
};

void CcidConnection::detectContactless()
{
    transact(Vector(std::span { kGetUidCommand }), [weakThis = WeakPtr { *this }] (Vector<uint8_t>&& response) mutable {
        ASSERT(RunLoop::isMain());
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;
        // Only contactless smart cards have uid, check for longer length than apdu status
        if (response.size() > 2)
            protectedThis->m_contactless = true;
    });
}

void CcidConnection::trySelectFidoApplet()
{
    transact(Vector(std::span { kCtapNfcAppletSelectionCommand }), [weakThis = WeakPtr { *this }] (Vector<uint8_t>&& response) mutable {
        ASSERT(RunLoop::isMain());
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;
        if (equalSpans(response.span(), std::span { kCtapNfcAppletSelectionU2f })
            || equalSpans(response.span(), std::span { kCtapNfcAppletSelectionCtap })) {
            if (RefPtr service = protectedThis->m_service.get())
                service->didConnectTag();
            return;
        }
        protectedThis->transact(Vector(std::span { kCtapNfcAppletSelectionCommand }), [weakThis = WTFMove(weakThis)] (Vector<uint8_t>&& response) mutable {
            ASSERT(RunLoop::isMain());
            RefPtr protectedThis = weakThis.get();
            if (!protectedThis)
                return;
            if (equalSpans(response.span(), std::span { kCtapNfcAppletSelectionU2f })) {
                if (RefPtr service = protectedThis->m_service.get())
                    service->didConnectTag();
            }
        });
    });
}

void CcidConnection::transact(Vector<uint8_t>&& data, DataReceivedCallback&& callback) const
{
    [m_smartCard beginSessionWithReply:makeBlockPtr([this, data = WTFMove(data), callback = WTFMove(callback)] (BOOL success, NSError *error) mutable {
        if (!success)
            return;
        [m_smartCard transmitRequest:toNSData(data).autorelease() reply:makeBlockPtr([this, callback = WTFMove(callback)](NSData * _Nullable nsResponse, NSError * _Nullable error) mutable {
            [m_smartCard endSession];
            callOnMainRunLoop([response = makeVector(nsResponse), callback = WTFMove(callback)] () mutable {
                callback(WTFMove(response));
            });
        }).get()];
    }).get()];
}


void CcidConnection::stop() const
{
}

// NearField polling is a one shot polling. It halts after tags are detected.
// Therefore, a restart process is needed to resume polling after error.
void CcidConnection::restartPolling()
{
    m_retryTimer.startOneShot(1_s); // Magic number to give users enough time for reactions.
}

void CcidConnection::startPolling()
{
    detectContactless();
    trySelectFidoApplet();
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
