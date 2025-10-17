/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#include "PluginInfoProvider.h"

#include "FrameLoader.h"
#include "LocalFrame.h"
#include "Page.h"
#include "SubframeLoader.h"

namespace WebCore {

PluginInfoProvider::~PluginInfoProvider()
{
    ASSERT(m_pages.isEmptyIgnoringNullReferences());
}

void PluginInfoProvider::clearPagesPluginData()
{
    for (auto& page : m_pages)
        page.clearPluginData();
}

void PluginInfoProvider::refresh(bool reloadPages)
{
    refreshPlugins();

    Vector<Ref<LocalFrame>> framesNeedingReload;

    for (auto& page : m_pages) {
        page.clearPluginData();

        if (!reloadPages)
            continue;

        for (Frame* frame = &page.mainFrame(); frame; frame = frame->tree().traverseNext()) {
            auto* localFrame = dynamicDowncast<LocalFrame>(frame);
            if (!localFrame)
                continue;
            if (localFrame->loader().subframeLoader().containsPlugins()) {
                if (RefPtr localMainFrame = page.localMainFrame())
                    framesNeedingReload.append(*localMainFrame);
            }
        }
    }

    for (Ref frame : framesNeedingReload)
        frame->protectedLoader()->reload();
}

void PluginInfoProvider::addPage(Page& page)
{
    ASSERT(!m_pages.contains(page));

    m_pages.add(page);
}

void PluginInfoProvider::removePage(Page& page)
{
    ASSERT(m_pages.contains(page));

    m_pages.remove(page);
}

}
