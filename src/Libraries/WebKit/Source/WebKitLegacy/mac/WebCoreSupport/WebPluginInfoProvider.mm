/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#import "WebPluginInfoProvider.h"

#import "WebPluginDatabase.h"
#import "WebPluginPackage.h"
#import <WebCore/FrameLoader.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/Page.h>
#import <wtf/BlockObjCExceptions.h>

using namespace WebCore;

WebPluginInfoProvider& WebPluginInfoProvider::singleton()
{
    static WebPluginInfoProvider& pluginInfoProvider = adoptRef(*new WebPluginInfoProvider).leakRef();

    return pluginInfoProvider;
}

WebPluginInfoProvider::WebPluginInfoProvider()
{
}

WebPluginInfoProvider::~WebPluginInfoProvider()
{
}

void WebPluginInfoProvider::refreshPlugins()
{
    [[WebPluginDatabase sharedDatabaseIfExists] refresh];
}

Vector<WebCore::PluginInfo> WebPluginInfoProvider::pluginInfo(WebCore::Page& page, std::optional<Vector<SupportedPluginIdentifier>>&)
{
    Vector<WebCore::PluginInfo> plugins;

    BEGIN_BLOCK_OBJC_EXCEPTIONS


    // WebKit1 has no application plug-ins, so we don't need to add them here.
    auto* localMainFrame = dynamicDowncast<LocalFrame>(page.mainFrame());
    if (!localMainFrame)
        return plugins;

    return plugins;

    END_BLOCK_OBJC_EXCEPTIONS

    return plugins;
}

Vector<WebCore::PluginInfo> WebPluginInfoProvider::webVisiblePluginInfo(WebCore::Page& page, const URL&)
{
    std::optional<Vector<SupportedPluginIdentifier>> supportedPluginIdentifiers;
    return pluginInfo(page, supportedPluginIdentifiers);
}
