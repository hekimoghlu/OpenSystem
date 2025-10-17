/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#include "FrameProcess.h"

#include "BrowsingContextGroup.h"
#include "WebPageProxy.h"
#include "WebPreferences.h"
#include "WebProcessProxy.h"
#include <WebCore/Site.h>

namespace WebKit {

FrameProcess::FrameProcess(WebProcessProxy& process, BrowsingContextGroup& group, const WebCore::Site& site, const WebPreferences& preferences)
    : m_process(process)
    , m_browsingContextGroup(group)
    , m_site(site)
{
    if (preferences.siteIsolationEnabled()) {
        group.addFrameProcess(*this);
        process.didStartUsingProcessForSiteIsolation(site);
    } else
        m_browsingContextGroup = nullptr;
}

FrameProcess::~FrameProcess()
{
    if (m_browsingContextGroup)
        m_browsingContextGroup->removeFrameProcess(*this);
}

}
