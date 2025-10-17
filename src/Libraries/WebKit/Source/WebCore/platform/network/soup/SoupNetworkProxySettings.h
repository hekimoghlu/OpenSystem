/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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

#include <wtf/HashMap.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

namespace WebCore {

enum class SoupNetworkProxySettingsMode : uint8_t {
    Default,
    NoProxy,
    Custom,
    Auto
};

struct SoupNetworkProxySettings {
    using Mode = SoupNetworkProxySettingsMode;

    SoupNetworkProxySettings() = default;

    explicit SoupNetworkProxySettings(Mode proxyMode)
        : mode(proxyMode)
    {
    }

    SoupNetworkProxySettings(Mode proxyMode, const CString& defaultURL, const GUniquePtr<char*>& hosts, const HashMap<CString, CString>& map)
        : mode(proxyMode)
        , defaultProxyURL(defaultURL)
        , ignoreHosts(g_strdupv(hosts.get()))
        , proxyMap(map)
    {
    }

    SoupNetworkProxySettings(const WebCore::SoupNetworkProxySettings& other)
        : mode(other.mode)
        , defaultProxyURL(other.defaultProxyURL)
        , ignoreHosts(g_strdupv(other.ignoreHosts.get()))
        , proxyMap(other.proxyMap)
    {
    }

    SoupNetworkProxySettings& operator=(const WebCore::SoupNetworkProxySettings& other)
    {
        mode = other.mode;
        defaultProxyURL = other.defaultProxyURL;
        ignoreHosts.reset(g_strdupv(other.ignoreHosts.get()));
        proxyMap = other.proxyMap;
        return *this;
    }

    bool isEmpty() const
    {
        switch (mode) {
        case Mode::Default:
        case Mode::NoProxy:
            return false;
        case Mode::Custom:
            return defaultProxyURL.isNull() && !ignoreHosts && proxyMap.isEmpty();
        case Mode::Auto:
            return defaultProxyURL.isNull();
        }
        RELEASE_ASSERT_NOT_REACHED();
    }

    Mode mode { Mode::Default };
    CString defaultProxyURL;
    GUniquePtr<char*> ignoreHosts;
    HashMap<CString, CString> proxyMap;
};

} // namespace WebCore
