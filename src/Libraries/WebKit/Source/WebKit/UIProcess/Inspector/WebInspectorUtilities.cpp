/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
#include "WebInspectorUtilities.h"

#include "APIProcessPoolConfiguration.h"
#include "WebPageGroup.h"
#include "WebPageProxy.h"
#include "WebProcessPool.h"
#include "WebProcessProxy.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/MakeString.h>

#if PLATFORM(COCOA)
#include <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#endif

namespace WebKit {

typedef HashMap<WebPageProxy*, unsigned> PageLevelMap;

static PageLevelMap& pageLevelMap()
{
    static NeverDestroyed<PageLevelMap> map;
    return map;
}

unsigned inspectorLevelForPage(WebPageProxy* page)
{
    if (page) {
        auto findResult = pageLevelMap().find(page);
        if (findResult != pageLevelMap().end())
            return findResult->value + 1;
    }

    return 1;
}

String defaultInspectorPageGroupIdentifierForPage(WebPageProxy* page)
{
    return makeString("__WebInspectorPageGroupLevel"_s, inspectorLevelForPage(page), "__"_s);
}

void trackInspectorPage(WebPageProxy* inspectorPage, WebPageProxy* inspectedPage)
{
    pageLevelMap().set(inspectorPage, inspectorLevelForPage(inspectedPage));
}

void untrackInspectorPage(WebPageProxy* inspectorPage)
{
    pageLevelMap().remove(inspectorPage);
}

static WebProcessPool* s_mainInspectorProcessPool;
static WebProcessPool* s_nestedInspectorProcessPool;

static WeakHashSet<WebProcessPool>& allInspectorProcessPools()
{
    static NeverDestroyed<WeakHashSet<WebProcessPool>> allInspectorProcessPools;
    return allInspectorProcessPools.get();
}

WebProcessPool& defaultInspectorProcessPool(unsigned inspectionLevel)
{
    // Having our own process pool removes us from the main process pool and
    // guarantees no process sharing for our user interface.
    WebProcessPool*& pool = (inspectionLevel == 1) ? s_mainInspectorProcessPool : s_nestedInspectorProcessPool;
    if (!pool) {
        auto configuration = API::ProcessPoolConfiguration::create();
        pool = &WebProcessPool::create(configuration.get()).leakRef();
        prepareProcessPoolForInspector(*pool);
    }
    return *pool;
}

void prepareProcessPoolForInspector(WebProcessPool& processPool)
{
    allInspectorProcessPools().add(processPool);
}

bool isInspectorProcessPool(WebProcessPool& processPool)
{
    return allInspectorProcessPools().contains(processPool);
}

bool isInspectorPage(WebPageProxy& webPage)
{
    return pageLevelMap().contains(&webPage);
}

#if PLATFORM(COCOA)
CFStringRef bundleIdentifierForSandboxBroker()
{
    if (applicationBundleIdentifier() == "com.apple.SafariTechnologyPreview"_s)
        return CFSTR("com.apple.SafariTechnologyPreview.SandboxBroker");
    if (applicationBundleIdentifier() == "com.apple.Safari.automation"_s)
        return CFSTR("com.apple.Safari.automation.SandboxBroker");

    return CFSTR("com.apple.Safari.SandboxBroker");
}
#endif // PLATFORM(COCOA)

} // namespace WebKit
