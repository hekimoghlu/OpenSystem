/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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

#include "LegacyCustomProtocolID.h"
#include "MessageReceiver.h"
#include "NetworkProcessSupplement.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
OBJC_CLASS NSURLSessionConfiguration;
OBJC_CLASS WKCustomProtocol;
#endif

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
} // namespace WebCore

namespace WebKit {

enum class CacheStoragePolicy : uint8_t;
class NetworkProcess;
struct NetworkProcessCreationParameters;

class LegacyCustomProtocolManager : public NetworkProcessSupplement, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(LegacyCustomProtocolManager);
    WTF_MAKE_NONCOPYABLE(LegacyCustomProtocolManager);
public:
    explicit LegacyCustomProtocolManager(NetworkProcess&);

    static ASCIILiteral supplementName();

    void registerScheme(const String&);
    void unregisterScheme(const String&);
    bool supportsScheme(const String&);

    void ref() const final;
    void deref() const final;

#if PLATFORM(COCOA)
    typedef RetainPtr<WKCustomProtocol> CustomProtocol;
#endif

    LegacyCustomProtocolID addCustomProtocol(CustomProtocol&&);
    void removeCustomProtocol(LegacyCustomProtocolID);
    void startLoading(LegacyCustomProtocolID, const WebCore::ResourceRequest&);
    void stopLoading(LegacyCustomProtocolID);

#if PLATFORM(COCOA)
    void registerProtocolClass(NSURLSessionConfiguration*);
    static void networkProcessCreated(NetworkProcess&);
#endif

private:
    // NetworkProcessSupplement
    void initialize(const NetworkProcessCreationParameters&) override;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void didFailWithError(LegacyCustomProtocolID, const WebCore::ResourceError&);
    void didLoadData(LegacyCustomProtocolID, std::span<const uint8_t>);
    void didReceiveResponse(LegacyCustomProtocolID, const WebCore::ResourceResponse&, CacheStoragePolicy);
    void didFinishLoading(LegacyCustomProtocolID);
    void wasRedirectedToRequest(LegacyCustomProtocolID, const WebCore::ResourceRequest&, const WebCore::ResourceResponse& redirectResponse);

    void registerProtocolClass();
    Ref<NetworkProcess> protectedNetworkProcess() const;

    CheckedRef<NetworkProcess> m_networkProcess;

    typedef HashMap<LegacyCustomProtocolID, CustomProtocol> CustomProtocolMap;
    CustomProtocolMap m_customProtocolMap WTF_GUARDED_BY_LOCK(m_customProtocolMapLock);
    Lock m_customProtocolMapLock;

#if PLATFORM(COCOA)
    HashSet<String, ASCIICaseInsensitiveHash> m_registeredSchemes WTF_GUARDED_BY_LOCK(m_registeredSchemesLock);
    Lock m_registeredSchemesLock;

    // WKCustomProtocol objects can be removed from the m_customProtocolMap from multiple threads.
    // We return a RetainPtr here because it is unsafe to return a raw pointer since the object might immediately be destroyed from a different thread.
    RetainPtr<WKCustomProtocol> protocolForID(LegacyCustomProtocolID);
#endif
};

} // namespace WebKit
