/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
#include "WebPluginInfoProvider.h"

#include <WebCore/Page.h>

#if ENABLE(PDF_PLUGIN)
#include "PDFPluginBase.h"
#endif

namespace WebKit {

WebPluginInfoProvider& WebPluginInfoProvider::singleton()
{
    static auto& pluginInfoProvider = adoptRef(*new WebPluginInfoProvider).leakRef();
    return pluginInfoProvider;
}

void WebPluginInfoProvider::refreshPlugins()
{
}

static Vector<WebCore::PluginInfo> pluginInfoVector(WebCore::Page& page)
{
#if ENABLE(PDF_PLUGIN)
    auto& settings = page.settings();
    if (settings.unifiedPDFEnabled() || settings.pdfPluginEnabled())
        return { PDFPluginBase::pluginInfo() };
#endif
    return { };
}

Vector<WebCore::PluginInfo> WebPluginInfoProvider::pluginInfo(WebCore::Page& page, std::optional<Vector<WebCore::SupportedPluginIdentifier>>&)
{
    return pluginInfoVector(page);
}

Vector<WebCore::PluginInfo> WebPluginInfoProvider::webVisiblePluginInfo(WebCore::Page& page, const URL&)
{
    return pluginInfoVector(page);
}

}
