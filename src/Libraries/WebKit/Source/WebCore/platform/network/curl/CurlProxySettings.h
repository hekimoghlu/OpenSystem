/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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

#include <wtf/ArgumentCoder.h>
#include <wtf/URL.h>
#include <wtf/text/WTFString.h>

#define kCURLAUTH_DIGEST_IE    (((unsigned long) 1) << 4)
#define kCURLAUTH_ANY          (~kCURLAUTH_DIGEST_IE)

namespace WebCore {

class CurlProxySettings {
public:
    enum class Mode : uint8_t {
        Default,
        NoProxy,
        Custom
    };

    struct DefaultData {
    };
    struct NoProxyData {
    };
    struct CustomData {
        URL url;
        String ignoreHosts;
    };

    CurlProxySettings() = default;
    WEBCORE_EXPORT explicit CurlProxySettings(Mode mode) : m_mode(mode) { }
    WEBCORE_EXPORT CurlProxySettings(URL&&, String&& ignoreHosts);

    WEBCORE_EXPORT CurlProxySettings(const CurlProxySettings&) = default;
    WEBCORE_EXPORT CurlProxySettings& operator=(const CurlProxySettings&) = default;

    bool isEmpty() const { return m_mode == Mode::Custom && m_urlSerializedWithPort.isEmpty() && m_ignoreHosts.isEmpty(); }

    Mode mode() const { return m_mode; }
    const String& url() const { return m_urlSerializedWithPort; }
    const String& ignoreHosts() const { return m_ignoreHosts; }

    WEBCORE_EXPORT void setUserPass(const String&, const String&);
    const String user() const { return m_url.user(); }
    const String password() const { return m_url.password(); }

    void setDefaultAuthMethod() { m_authMethod = kCURLAUTH_ANY; }
    void setAuthMethod(long);
    long authMethod() const { return m_authMethod; }

private:
    friend struct IPC::ArgumentCoder<CurlProxySettings, void>;
    using IPCData = std::variant<DefaultData, NoProxyData, CustomData>;
    WEBCORE_EXPORT IPCData toIPCData() const;
    WEBCORE_EXPORT static CurlProxySettings fromIPCData(IPCData&&);

    Mode m_mode { Mode::Default };
    URL m_url;
    String m_ignoreHosts;
    long m_authMethod { static_cast<long>(kCURLAUTH_ANY) };

    // We can't simply use m_url.string() because we need to explicitly indicate the port number
    // to libcurl. URLParser omit the default port while parsing, but libcurl assume 1080 as a
    // default HTTP Proxy, not 80, if port number is not in the url.
    String m_urlSerializedWithPort;

    void rebuildUrl();
};

bool protocolIsInSocksFamily(const URL&);

} // namespace WebCore
