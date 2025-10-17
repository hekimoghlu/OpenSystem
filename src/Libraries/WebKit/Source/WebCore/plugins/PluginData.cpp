/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "PluginData.h"

#include "Document.h"
#include "LocalFrame.h"
#include "LocalizedStrings.h"
#include "Page.h"
#include "PluginInfoProvider.h"

namespace WebCore {

#if PLATFORM(COCOA)
static inline bool isBuiltInPDFPlugIn(const PluginInfo& plugin)
{
    return equalLettersIgnoringASCIICase(plugin.bundleIdentifier, "com.apple.webkit.builtinpdfplugin"_s);
}
#else
static inline bool isBuiltInPDFPlugIn(const PluginInfo&)
{
    return false;
}
#endif

PluginData::PluginData(Page& page)
    : m_page(page)
{
    initPlugins();
}

void PluginData::initPlugins()
{
    ASSERT(m_plugins.isEmpty());
    m_plugins = m_page.pluginInfoProvider().pluginInfo(m_page, m_supportedPluginIdentifiers);

    for (auto& plugin : m_plugins) {
        if (isBuiltInPDFPlugIn(plugin)) {
            m_builtInPDFPluginInfo = plugin;
            break;
        }
    }
}

const Vector<PluginInfo>& PluginData::webVisiblePlugins() const
{
    auto* localMainFrame = dynamicDowncast<LocalFrame>(m_page.mainFrame());
    auto documentURL = localMainFrame && localMainFrame->document() ? localMainFrame->document()->url() : URL { };
    if (!documentURL.isNull() && !protocolHostAndPortAreEqual(m_cachedVisiblePlugins.pageURL, documentURL)) {
        m_cachedVisiblePlugins.pageURL = WTFMove(documentURL);
        m_cachedVisiblePlugins.pluginList = std::nullopt;
    }

    if (!m_cachedVisiblePlugins.pluginList)
        m_cachedVisiblePlugins.pluginList = m_page.pluginInfoProvider().webVisiblePluginInfo(m_page, m_cachedVisiblePlugins.pageURL);

    return *m_cachedVisiblePlugins.pluginList;
}

Vector<MimeClassInfo> PluginData::webVisibleMimeTypes() const
{
    Vector<MimeClassInfo> result;
    for (auto& plugin : webVisiblePlugins())
        result.appendVector(plugin.mimes);
    return result;
}

static bool supportsMimeTypeForPlugins(const String& mimeType, const PluginData::AllowedPluginTypes allowedPluginTypes, const Vector<PluginInfo>& plugins)
{
    for (auto& plugin : plugins) {
        for (auto& type : plugin.mimes) {
            if (type.type == mimeType && (allowedPluginTypes == PluginData::AllPlugins || plugin.isApplicationPlugin))
                return true;
        }
    }
    return false;
}

bool PluginData::supportsMimeType(const String& mimeType, const AllowedPluginTypes allowedPluginTypes) const
{
    return supportsMimeTypeForPlugins(mimeType, allowedPluginTypes, plugins());
}

bool PluginData::supportsWebVisibleMimeType(const String& mimeType, const AllowedPluginTypes allowedPluginTypes) const
{
    return supportsMimeTypeForPlugins(mimeType, allowedPluginTypes, webVisiblePlugins());
}

bool PluginData::supportsWebVisibleMimeTypeForURL(const String& mimeType, const AllowedPluginTypes allowedPluginTypes, const URL& url) const
{
    if (!protocolHostAndPortAreEqual(m_cachedVisiblePlugins.pageURL, url))
        m_cachedVisiblePlugins = { url, m_page.pluginInfoProvider().webVisiblePluginInfo(m_page, url) };
    if (!m_cachedVisiblePlugins.pluginList)
        return false;
    return supportsMimeTypeForPlugins(mimeType, allowedPluginTypes, *m_cachedVisiblePlugins.pluginList);
}

String PluginData::pluginFileForWebVisibleMimeType(const String& mimeType) const
{
    for (auto& plugin : webVisiblePlugins()) {
        for (auto& type : plugin.mimes) {
            if (type.type == mimeType)
                return plugin.file;
        }
    }
    return { };
}

PluginInfo PluginData::dummyPDFPluginInfo()
{
    PluginInfo info;

    info.name = "Dummy Plugin"_s;
    info.desc = pdfDocumentTypeDescription();
    info.file = "internal-pdf-viewer"_s;
    info.isApplicationPlugin = true;
    info.clientLoadPolicy = PluginLoadClientPolicy::Undefined;

    MimeClassInfo pdfMimeClassInfo;
    pdfMimeClassInfo.type = "application/pdf"_s;
    pdfMimeClassInfo.desc = pdfDocumentTypeDescription();
    pdfMimeClassInfo.extensions.append("pdf"_s);
    info.mimes.append(pdfMimeClassInfo);

    MimeClassInfo textPDFMimeClassInfo;
    textPDFMimeClassInfo.type = "text/pdf"_s;
    textPDFMimeClassInfo.desc = pdfDocumentTypeDescription();
    textPDFMimeClassInfo.extensions.append("pdf"_s);
    info.mimes.append(textPDFMimeClassInfo);

    return info;
}

} // namespace WebCore
