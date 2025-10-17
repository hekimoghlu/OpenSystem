/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

#include <wtf/RefCounted.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Page;

enum class PluginLoadClientPolicy : uint8_t {
    // No client-specific plug-in load policy has been defined. The plug-in should be visible in navigator.plugins and WebKit should synchronously
    // ask the client whether the plug-in should be loaded.
    Undefined = 0,

    // The plug-in module should be blocked from being instantiated. The plug-in should be hidden in navigator.plugins.
    Block,

    // WebKit should synchronously ask the client whether the plug-in should be loaded. The plug-in should be visible in navigator.plugins.
    Ask,

    // The plug-in module may be loaded if WebKit is not blocking it.
    Allow,

    // The plug-in module should be loaded irrespective of whether WebKit has asked it to be blocked.
    AllowAlways,
};

struct MimeClassInfo {
    AtomString type;
    String desc;
    Vector<String> extensions;

    friend bool operator==(const MimeClassInfo&, const MimeClassInfo&) = default;
};

struct PluginInfo {
    String name;
    String file;
    String desc;
    Vector<MimeClassInfo> mimes;
    bool isApplicationPlugin { false };

    PluginLoadClientPolicy clientLoadPolicy { PluginLoadClientPolicy::Undefined };

    String bundleIdentifier;
#if PLATFORM(MAC)
    String versionString;
#endif

    friend bool operator==(const PluginInfo&, const PluginInfo&) = default;
};

struct SupportedPluginIdentifier {
    String matchingDomain;
    String pluginIdentifier;
};

// FIXME: merge with PluginDatabase in the future
class PluginData : public RefCounted<PluginData> {
public:
    static Ref<PluginData> create(Page& page) { return adoptRef(*new PluginData(page)); }

    const Vector<PluginInfo>& plugins() const { return m_plugins; }
    WEBCORE_EXPORT const Vector<PluginInfo>& webVisiblePlugins() const;
    WEBCORE_EXPORT Vector<MimeClassInfo> webVisibleMimeTypes() const;

    enum AllowedPluginTypes {
        AllPlugins,
        OnlyApplicationPlugins
    };
    WEBCORE_EXPORT bool supportsMimeType(const String& mimeType, const AllowedPluginTypes) const;
    WEBCORE_EXPORT bool supportsWebVisibleMimeType(const String& mimeType, const AllowedPluginTypes) const;
    WEBCORE_EXPORT bool supportsWebVisibleMimeTypeForURL(const String& mimeType, const AllowedPluginTypes, const URL&) const;

    String pluginFileForWebVisibleMimeType(const String& mimeType) const;

    const std::optional<PluginInfo>& builtInPDFPlugin() const { return m_builtInPDFPluginInfo; }

    static PluginInfo dummyPDFPluginInfo();

private:
    explicit PluginData(Page&);
    void initPlugins();

protected:
    Page& m_page;
    Vector<PluginInfo> m_plugins;
    std::optional<Vector<SupportedPluginIdentifier>> m_supportedPluginIdentifiers;

    struct CachedVisiblePlugins {
        URL pageURL;
        std::optional<Vector<PluginInfo>> pluginList;
    };
    mutable CachedVisiblePlugins m_cachedVisiblePlugins;
    std::optional<PluginInfo> m_builtInPDFPluginInfo;
};

inline bool isSupportedPlugin(const Vector<SupportedPluginIdentifier>& pluginIdentifiers, const URL& pageURL, const String& pluginIdentifier)
{
    return pluginIdentifiers.findIf([&] (auto&& plugin) {
        return pageURL.isMatchingDomain(plugin.matchingDomain) && plugin.pluginIdentifier == pluginIdentifier;
    }) != notFound;
}

} // namespace WebCore
