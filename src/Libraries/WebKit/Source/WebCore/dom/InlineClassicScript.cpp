/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#include "InlineClassicScript.h"

#include "ElementInlines.h"
#include "HTMLNames.h"
#include "ScriptElement.h"

namespace WebCore {

Ref<InlineClassicScript> InlineClassicScript::create(ScriptElement& scriptElement)
{
    Ref element = scriptElement.element();
    return adoptRef(*new InlineClassicScript(
        element->nonce(),
        element->attributeWithoutSynchronization(HTMLNames::crossoriginAttr),
        scriptElement.scriptCharset(),
        element->localName(),
        element->isInUserAgentShadowTree()));
}

InlineClassicScript::InlineClassicScript(const AtomString& nonce, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree)
    : ScriptElementCachedScriptFetcher(nonce, ReferrerPolicy::EmptyString, RequestPriority::Auto, crossOriginMode, charset, initiatorType, isInUserAgentShadowTree)
{
}

}
