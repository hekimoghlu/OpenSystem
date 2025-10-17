/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#include "WebMDNSRegister.h"

#if ENABLE(WEB_RTC)

#include "LibWebRTCNetwork.h"
#include "NetworkMDNSRegisterMessages.h"
#include "NetworkProcessConnection.h"
#include "WebProcess.h"
#include <WebCore/Document.h>

namespace WebKit {
using namespace WebCore;

WebMDNSRegister::WebMDNSRegister(LibWebRTCNetwork& libWebRTCNetwork)
    : m_libWebRTCNetwork(libWebRTCNetwork)
{
}

void WebMDNSRegister::ref() const
{
    m_libWebRTCNetwork->ref();
}

void WebMDNSRegister::deref() const
{
    m_libWebRTCNetwork->deref();
}

void WebMDNSRegister::finishedRegisteringMDNSName(WebCore::ScriptExecutionContextIdentifier documentIdentifier, const String& ipAddress, String&& name, std::optional<MDNSRegisterError> error, CompletionHandler<void(const String&, std::optional<MDNSRegisterError>)>&& completionHandler)
{
    if (!error) {
        auto iterator = m_registeringDocuments.find(documentIdentifier);
        if (iterator == m_registeringDocuments.end())
            return completionHandler(name, WebCore::MDNSRegisterError::DNSSD);
        iterator->value.add(ipAddress, name);
    }

    completionHandler(name, error);
}

void WebMDNSRegister::unregisterMDNSNames(ScriptExecutionContextIdentifier identifier)
{
    if (m_registeringDocuments.take(identifier).isEmpty())
        return;

    Ref connection = WebProcess::singleton().ensureNetworkProcessConnection().connection();
    connection->send(Messages::NetworkMDNSRegister::UnregisterMDNSNames { identifier }, 0);
}

void WebMDNSRegister::registerMDNSName(ScriptExecutionContextIdentifier identifier, const String& ipAddress, CompletionHandler<void(const String&, std::optional<MDNSRegisterError>)>&& callback)
{
    auto& map = m_registeringDocuments.ensure(identifier, [] {
        return HashMap<String, String> { };
    }).iterator->value;

    auto iterator = map.find(ipAddress);
    if (iterator != map.end()) {
        callback(iterator->value, { });
        return;
    }

    auto& connection = WebProcess::singleton().ensureNetworkProcessConnection().connection();
    connection.sendWithAsyncReply(Messages::NetworkMDNSRegister::RegisterMDNSName { identifier, ipAddress }, [weakThis = WeakPtr { *this }, callback = WTFMove(callback), identifier, ipAddress] (String&& mdnsName, std::optional<MDNSRegisterError> error) mutable {
        if (RefPtr protectedThis = weakThis.get())
            protectedThis->finishedRegisteringMDNSName(identifier, ipAddress, WTFMove(mdnsName), error, WTFMove(callback));
        else
            callback({ }, MDNSRegisterError::Internal);
    });
}

} // namespace WebKit

#endif // ENABLE(WEB_RTC)
