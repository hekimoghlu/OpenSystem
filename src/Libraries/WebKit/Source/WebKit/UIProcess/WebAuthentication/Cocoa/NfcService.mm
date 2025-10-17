/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
#import "NfcService.h"

#if ENABLE(WEB_AUTHN)

#import "CtapNfcDriver.h"
#import "NearFieldSPI.h"
#import "NfcConnection.h"
#import <wtf/BlockPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/RunLoop.h>

#import "NearFieldSoftLink.h"

namespace WebKit {

Ref<NfcService> NfcService::create(AuthenticatorTransportServiceObserver& observer)
{
    return adoptRef(*new NfcService(observer));
}

NfcService::NfcService(AuthenticatorTransportServiceObserver& observer)
    : FidoService(observer)
    , m_restartTimer(RunLoop::main(), this, &NfcService::platformStartDiscovery)
{
}

NfcService::~NfcService() = default;

bool NfcService::isAvailable()
{
#if HAVE(NEAR_FIELD)
    return [[getNFHardwareManagerClass() sharedHardwareManager] areFeaturesSupported:NFFeatureReaderMode outError:nil];
#else
    return false;
#endif
}

void NfcService::didConnectTag()
{
#if HAVE(NEAR_FIELD)
    auto connection = m_connection;
    ASSERT(connection);
    getInfo(CtapNfcDriver::create(connection.releaseNonNull()));
#endif
}

void NfcService::didDetectMultipleTags() const
{
    if (auto* observer = this->observer())
        observer->serviceStatusUpdated(WebAuthenticationStatus::MultipleNFCTagsPresent);
}

#if HAVE(NEAR_FIELD)
void NfcService::setConnection(Ref<NfcConnection>&& connection)
{
    m_connection = WTFMove(connection);
}
#endif

void NfcService::startDiscoveryInternal()
{
    platformStartDiscovery();
}

void NfcService::restartDiscoveryInternal()
{
#if HAVE(NEAR_FIELD)
    if (m_connection)
        m_connection->stop();
#endif
    m_restartTimer.startOneShot(1_s); // Magic number to give users enough time for reactions.
}

void NfcService::platformStartDiscovery()
{
#if HAVE(NEAR_FIELD)
    if (!isAvailable())
        return;

    // Will be executed in a different thread.
    auto callback = makeBlockPtr([weakThis = WeakPtr { *this }, this] (NFReaderSession *session, NSError *error) mutable {
        ASSERT(!RunLoop::isMain());
        if (error) {
            LOG_ERROR("Couldn't start a NFC reader session: %@", error);
            return;
        }

        RunLoop::main().dispatch([weakThis = WTFMove(weakThis), this, session = retainPtr(session)] () mutable {
            if (!weakThis) {
                [session endSession];
                return;
            }

            // NfcConnection will take care of polling tags and connecting to them.
            m_connection = NfcConnection::create(WTFMove(session), *this);
        });
    });
    [[getNFHardwareManagerClass() sharedHardwareManager] startReaderSession:callback.get()];
#endif // HAVE(NEAR_FIELD)
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
