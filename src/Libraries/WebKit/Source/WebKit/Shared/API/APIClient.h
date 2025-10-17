/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#ifndef APIClient_h
#define APIClient_h

#include <algorithm>
#include <array>
#include <tuple>
#include <wtf/StdLibExtras.h>

namespace API {

template<typename ClientInterface> struct ClientTraits;

template<typename ClientInterface> class Client {
    typedef typename ClientTraits<ClientInterface>::Versions ClientVersions;
    static const int latestClientVersion = std::tuple_size<ClientVersions>::value - 1;
    typedef typename std::tuple_element<latestClientVersion, ClientVersions>::type LatestClientInterface;

    // Helper class that can return an std::array of element sizes in a tuple.
    template<typename> struct InterfaceSizes;
    template<typename... Interfaces> struct InterfaceSizes<std::tuple<Interfaces...>> {
        static std::array<size_t, sizeof...(Interfaces)> sizes()
        {
            return { { sizeof(Interfaces)... } };
        }
    };

public:
    Client()
    {
#if ASSERT_ENABLED
        auto interfaceSizes = InterfaceSizes<ClientVersions>::sizes();
        ASSERT(std::is_sorted(interfaceSizes.begin(), interfaceSizes.end()));
#endif

        initialize(nullptr);
    }

    void initialize(const ClientInterface* client)
    {
        if (client && client->version == latestClientVersion) {
            m_client = *reinterpret_cast<const LatestClientInterface*>(client);
            return;
        }

        zeroBytes(m_client);

        if (client && client->version < latestClientVersion) {
            auto interfaceSizes = InterfaceSizes<ClientVersions>::sizes();

            memcpySpan(asMutableByteSpan(m_client), unsafeMakeSpan(reinterpret_cast<const uint8_t*>(client), interfaceSizes[client->version]));
        }
    }

    const LatestClientInterface& client() const { return m_client; }

protected:
    LatestClientInterface m_client;
};

} // namespace API

#endif // APIClient_h
