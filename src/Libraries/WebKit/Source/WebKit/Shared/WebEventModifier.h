/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#pragma once

#include <wtf/Forward.h>

namespace WebCore {
class NavigationAction;
enum class PlatformEventModifier : uint8_t;
}

namespace WebKit {

enum class WebEventModifier : uint8_t {
    ShiftKey    = 1 << 0,
    ControlKey  = 1 << 1,
    AltKey      = 1 << 2,
    MetaKey     = 1 << 3,
    CapsLockKey = 1 << 4,
};

OptionSet<WebEventModifier> modifiersFromPlatformEventModifiers(OptionSet<WebCore::PlatformEventModifier>);
OptionSet<WebEventModifier> modifiersForNavigationAction(const WebCore::NavigationAction&);

} // namespace WebKit
