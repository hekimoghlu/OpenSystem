/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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

#if USE(LIBWEBRTC)

#include "LibWebRTCResolverIdentifier.h"
#include "RTCNetwork.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class LibWebRTCSocketFactory;

class WebRTCResolver : public RefCounted<WebRTCResolver> {
    WTF_MAKE_TZONE_ALLOCATED(WebRTCResolver);
public:
    static Ref<WebRTCResolver> create(LibWebRTCSocketFactory&, LibWebRTCResolverIdentifier);
    ~WebRTCResolver();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

private:
    WebRTCResolver(LibWebRTCSocketFactory&, LibWebRTCResolverIdentifier);

    void setResolvedAddress(const Vector<RTCNetwork::IPAddress>&);
    void resolvedAddressError(int);

    CheckedRef<LibWebRTCSocketFactory> m_socketFactory;
    LibWebRTCResolverIdentifier m_identifier;
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)
