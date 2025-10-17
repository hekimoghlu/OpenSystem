/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

enum class WebExtensionMenuItemContextType : uint16_t {
    Action    =  1 << 0,
    Audio     =  1 << 1,
    Editable  =  1 << 2,
    Frame     =  1 << 3,
    Image     =  1 << 4,
    Link      =  1 << 5,
    Page      =  1 << 6,
    Selection =  1 << 7,
    Tab       =  1 << 8,
    Video     =  1 << 9,
};

static constexpr OptionSet<WebExtensionMenuItemContextType> allWebExtensionMenuItemContextTypes()
{
    return {
        WebExtensionMenuItemContextType::Action,
        WebExtensionMenuItemContextType::Audio,
        WebExtensionMenuItemContextType::Editable,
        WebExtensionMenuItemContextType::Frame,
        WebExtensionMenuItemContextType::Image,
        WebExtensionMenuItemContextType::Link,
        WebExtensionMenuItemContextType::Page,
        WebExtensionMenuItemContextType::Selection,
        WebExtensionMenuItemContextType::Tab,
        WebExtensionMenuItemContextType::Video
    };
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
