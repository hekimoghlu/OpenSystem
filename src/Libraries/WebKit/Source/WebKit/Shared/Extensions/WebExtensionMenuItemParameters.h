/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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

#include "WebExtensionMenuItemContextType.h"
#include "WebExtensionMenuItemType.h"
#include <wtf/Forward.h>

namespace WebKit {

static constexpr size_t webExtensionActionMenuItemTopLevelLimit = 6;

struct WebExtensionMenuItemParameters {
    String identifier;
    std::optional<String> parentIdentifier;

    std::optional<WebExtensionMenuItemType> type;

    String title;
    String command;

    String iconsJSON;

    std::optional<bool> checked;
    std::optional<bool> enabled;
    std::optional<bool> visible;

    std::optional<Vector<String>> documentURLPatterns;
    std::optional<Vector<String>> targetURLPatterns;

    std::optional<OptionSet<WebExtensionMenuItemContextType>> contexts;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
