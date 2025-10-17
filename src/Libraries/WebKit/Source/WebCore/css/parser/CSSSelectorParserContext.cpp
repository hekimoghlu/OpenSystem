/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#include "CSSSelectorParserContext.h"

#include "CSSParserContext.h"
#include "DocumentInlines.h"
#include "Quirks.h"
#include <wtf/Hasher.h>

namespace WebCore {

CSSSelectorParserContext::CSSSelectorParserContext(const CSSParserContext& context)
    : mode(context.mode)
    , cssNestingEnabled(context.cssNestingEnabled)
#if ENABLE(SERVICE_CONTROLS)
    , imageControlsEnabled(context.imageControlsEnabled)
#endif
    , popoverAttributeEnabled(context.popoverAttributeEnabled)
    , targetTextPseudoElementEnabled(context.targetTextPseudoElementEnabled)
    , thumbAndTrackPseudoElementsEnabled(context.thumbAndTrackPseudoElementsEnabled)
    , viewTransitionsEnabled(context.propertySettings.viewTransitionsEnabled)
    , viewTransitionClassesEnabled(viewTransitionsEnabled && context.propertySettings.viewTransitionClassesEnabled)
    , viewTransitionTypesEnabled(viewTransitionsEnabled && context.viewTransitionTypesEnabled)
    , webkitMediaTextTrackDisplayQuirkEnabled(context.webkitMediaTextTrackDisplayQuirkEnabled)
{
}

CSSSelectorParserContext::CSSSelectorParserContext(const Document& document)
    : mode(document.inQuirksMode() ? HTMLQuirksMode : HTMLStandardMode)
    , cssNestingEnabled(document.settings().cssNestingEnabled())
#if ENABLE(SERVICE_CONTROLS)
    , imageControlsEnabled(document.settings().imageControlsEnabled())
#endif
    , popoverAttributeEnabled(document.settings().popoverAttributeEnabled())
    , targetTextPseudoElementEnabled(document.settings().targetTextPseudoElementEnabled())
    , thumbAndTrackPseudoElementsEnabled(document.settings().thumbAndTrackPseudoElementsEnabled())
    , viewTransitionsEnabled(document.settings().viewTransitionsEnabled())
    , viewTransitionClassesEnabled(viewTransitionsEnabled && document.settings().viewTransitionClassesEnabled())
    , viewTransitionTypesEnabled(viewTransitionsEnabled && document.settings().viewTransitionTypesEnabled())
    , webkitMediaTextTrackDisplayQuirkEnabled(document.quirks().needsWebKitMediaTextTrackDisplayQuirk())
{
}

void add(Hasher& hasher, const CSSSelectorParserContext& context)
{
    add(hasher,
        context.mode,
        context.cssNestingEnabled,
#if ENABLE(SERVICE_CONTROLS)
        context.imageControlsEnabled,
#endif
        context.popoverAttributeEnabled,
        context.targetTextPseudoElementEnabled,
        context.thumbAndTrackPseudoElementsEnabled,
        context.viewTransitionsEnabled,
        context.viewTransitionClassesEnabled,
        context.viewTransitionTypesEnabled,
        context.webkitMediaTextTrackDisplayQuirkEnabled
    );
}

} // namespace WebCore
