/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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
#include "ScriptElementCachedScriptFetcher.h"

#include "Element.h"
#include "ScriptElement.h"

namespace WebCore {

const ASCIILiteral ScriptElementCachedScriptFetcher::defaultCrossOriginModeForModule { "anonymous"_s };

CachedResourceHandle<CachedScript> ScriptElementCachedScriptFetcher::requestModuleScript(Document& document, const URL& sourceURL, String&& integrity, std::optional<ServiceWorkersMode> serviceWorkersMode) const
{
    // https://html.spec.whatwg.org/multipage/urls-and-fetching.html#cors-settings-attributes
    // If the fetcher is not module script, credential mode is always "same-origin" ("anonymous").
    // This code is for dynamic module import (`import` operator).

    return requestScriptWithCache(document, sourceURL, isClassicScript() ? defaultCrossOriginModeForModule : m_crossOriginMode, WTFMove(integrity), { }, serviceWorkersMode);
}

}
