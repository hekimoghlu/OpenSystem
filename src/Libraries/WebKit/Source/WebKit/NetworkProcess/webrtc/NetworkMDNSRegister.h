/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

#if ENABLE(WEB_RTC)

#include "RTCNetwork.h"
#include <WebCore/ProcessQualified.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/Expected.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>

#if PLATFORM(COCOA) && defined __has_include && __has_include(<dns_sd.h>)
#define ENABLE_MDNS 1
#else
#define ENABLE_MDNS 0
#endif

#if ENABLE_MDNS
#include <dns_sd.h>
#endif

namespace IPC {
class Connection;
class Decoder;
}

namespace PAL {
class SessionID;
}

namespace WebCore {
enum class MDNSRegisterError : uint8_t;
}

namespace WebKit {

class NetworkConnectionToWebProcess;

class NetworkMDNSRegister {
public:
    NetworkMDNSRegister(NetworkConnectionToWebProcess&);
    ~NetworkMDNSRegister();

    void ref() const;
    void deref() const;

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

#if ENABLE_MDNS
    void closeAndForgetService(DNSServiceRef);
#endif

    bool hasRegisteredName(const String&) const;

private:
    void unregisterMDNSNames(WebCore::ScriptExecutionContextIdentifier);
    void registerMDNSName(WebCore::ScriptExecutionContextIdentifier, const String& ipAddress, CompletionHandler<void(const String&, std::optional<WebCore::MDNSRegisterError>)>&&);

    PAL::SessionID sessionID() const;

    WeakRef<NetworkConnectionToWebProcess> m_connection;
    HashSet<String> m_registeredNames;

    HashMap<WebCore::ScriptExecutionContextIdentifier, Vector<String>> m_perDocumentRegisteredNames;

#if ENABLE_MDNS
    struct DNSServiceDeallocator;
    HashMap<WebCore::ScriptExecutionContextIdentifier, std::unique_ptr<_DNSServiceRef_t, DNSServiceDeallocator>> m_services;
#endif
};

} // namespace WebKit

#endif // ENABLE(WEB_RTC)
