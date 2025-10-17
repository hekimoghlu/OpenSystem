/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
#include "WebEventModifier.h"

#include <WebCore/NavigationAction.h>
#include <WebCore/PlatformEvent.h>
#include <wtf/OptionSet.h>

namespace WebKit {

OptionSet<WebEventModifier> modifiersFromPlatformEventModifiers(OptionSet<WebCore::PlatformEventModifier> modifiers)
{
    OptionSet<WebEventModifier> result;
    if (modifiers.contains(WebCore::PlatformEventModifier::ShiftKey))
        result.add(WebEventModifier::ShiftKey);
    if (modifiers.contains(WebCore::PlatformEventModifier::ControlKey))
        result.add(WebEventModifier::ControlKey);
    if (modifiers.contains(WebCore::PlatformEventModifier::AltKey))
        result.add(WebEventModifier::AltKey);
    if (modifiers.contains(WebCore::PlatformEventModifier::MetaKey))
        result.add(WebEventModifier::MetaKey);
    if (modifiers.contains(WebCore::PlatformEventModifier::CapsLockKey))
        result.add(WebEventModifier::CapsLockKey);
    return result;
}

OptionSet<WebEventModifier> modifiersForNavigationAction(const WebCore::NavigationAction& navigationAction)
{
    OptionSet<WebEventModifier> modifiers;
    auto keyStateEventData = navigationAction.keyStateEventData();
    if (keyStateEventData && keyStateEventData->isTrusted) {
        if (keyStateEventData->shiftKey)
            modifiers.add(WebEventModifier::ShiftKey);
        if (keyStateEventData->ctrlKey)
            modifiers.add(WebEventModifier::ControlKey);
        if (keyStateEventData->altKey)
            modifiers.add(WebEventModifier::AltKey);
        if (keyStateEventData->metaKey)
            modifiers.add(WebEventModifier::MetaKey);
    }
    return modifiers;
}

} // namespace WebKit
