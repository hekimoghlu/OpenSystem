/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "WKBundleScriptWorld.h"

#include "InjectedBundleScriptWorld.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"

WKTypeID WKBundleScriptWorldGetTypeID()
{
    return WebKit::toAPI(WebKit::InjectedBundleScriptWorld::APIType);
}

WKBundleScriptWorldRef WKBundleScriptWorldCreateWorld()
{
    RefPtr<WebKit::InjectedBundleScriptWorld> world = WebKit::InjectedBundleScriptWorld::create();
    return toAPI(world.leakRef());
}

WKBundleScriptWorldRef WKBundleScriptWorldNormalWorld()
{
    return toAPI(&WebKit::InjectedBundleScriptWorld::normalWorld());
}

void WKBundleScriptWorldClearWrappers(WKBundleScriptWorldRef scriptWorldRef)
{
    WebKit::toImpl(scriptWorldRef)->clearWrappers();
}

void WKBundleScriptWorldMakeAllShadowRootsOpen(WKBundleScriptWorldRef scriptWorldRef)
{
    WebKit::toImpl(scriptWorldRef)->makeAllShadowRootsOpen();
}

void WKBundleScriptWorldDisableOverrideBuiltinsBehavior(WKBundleScriptWorldRef scriptWorldRef)
{
    WebKit::toImpl(scriptWorldRef)->disableOverrideBuiltinsBehavior();
}

WKStringRef WKBundleScriptWorldCopyName(WKBundleScriptWorldRef scriptWorldRef)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(scriptWorldRef)->name());
}
