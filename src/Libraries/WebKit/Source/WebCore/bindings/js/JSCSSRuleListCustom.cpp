/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
#include "JSCSSRuleList.h"

#include "CSSRule.h"
#include "CSSRuleList.h"
#include "CSSStyleSheet.h"
#include "JSCSSRuleCustom.h"
#include "JSStyleSheetCustom.h"


namespace WebCore {
using namespace JSC;

bool JSCSSRuleListOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    JSCSSRuleList* jsCSSRuleList = jsCast<JSCSSRuleList*>(handle.slot()->asCell());
    if (!jsCSSRuleList->hasCustomProperties())
        return false;

    if (CSSStyleSheet* styleSheet = jsCSSRuleList->wrapped().styleSheet()) {
        if (UNLIKELY(reason))
            *reason = "CSSStyleSheet is opaque root"_s;

        return containsWebCoreOpaqueRoot(visitor, styleSheet);
    }
    
    if (CSSRule* cssRule = jsCSSRuleList->wrapped().item(0)) {
        if (UNLIKELY(reason))
            *reason = "CSSRule is opaque root"_s;

        return containsWebCoreOpaqueRoot(visitor, cssRule);
    }
    return false;
}

}
