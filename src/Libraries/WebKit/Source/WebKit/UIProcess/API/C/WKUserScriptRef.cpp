/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#include "WKUserScriptRef.h"

#include "APIArray.h"
#include "APIUserScript.h"
#include "WKAPICast.h"

using namespace WebKit;

WKTypeID WKUserScriptGetTypeID()
{
    return toAPI(API::UserScript::APIType);
}

WKUserScriptRef WKUserScriptCreate(WKStringRef sourceRef, WKURLRef url, WKArrayRef includeURLPatterns, WKArrayRef excludeURLPatterns, _WKUserScriptInjectionTime injectionTime, bool forMainFrameOnly)
{
    auto baseURLString = toWTFString(url);
    auto allowlist = toImpl(includeURLPatterns);
    auto blocklist = toImpl(excludeURLPatterns);

    auto baseURL = baseURLString.isEmpty() ? aboutBlankURL() : URL(URL(), baseURLString);
    return toAPI(&API::UserScript::create(WebCore::UserScript { toWTFString(sourceRef), WTFMove(baseURL), allowlist ? allowlist->toStringVector() : Vector<String>(), blocklist ? blocklist->toStringVector() : Vector<String>(), toUserScriptInjectionTime(injectionTime), forMainFrameOnly ? WebCore::UserContentInjectedFrames::InjectInTopFrameOnly : WebCore::UserContentInjectedFrames::InjectInAllFrames }, API::ContentWorld::pageContentWorldSingleton()).leakRef());
}

WKUserScriptRef WKUserScriptCreateWithSource(WKStringRef sourceRef, _WKUserScriptInjectionTime injectionTime, bool forMainFrameOnly)
{
    return toAPI(&API::UserScript::create(WebCore::UserScript { toWTFString(sourceRef), { }, { }, { }, toUserScriptInjectionTime(injectionTime), forMainFrameOnly ? WebCore::UserContentInjectedFrames::InjectInTopFrameOnly : WebCore::UserContentInjectedFrames::InjectInAllFrames }, API::ContentWorld::pageContentWorldSingleton()).leakRef());
}

WKStringRef WKUserScriptCopySource(WKUserScriptRef userScriptRef)
{
    return toCopiedAPI(toImpl(userScriptRef)->userScript().source());
}

_WKUserScriptInjectionTime WKUserScriptGetInjectionTime(WKUserScriptRef userScriptRef)
{
    return toWKUserScriptInjectionTime(toImpl(userScriptRef)->userScript().injectionTime());
}

bool WKUserScriptGetMainFrameOnly(WKUserScriptRef userScriptRef)
{
    return toImpl(userScriptRef)->userScript().injectedFrames() == WebCore::UserContentInjectedFrames::InjectInTopFrameOnly;
}
