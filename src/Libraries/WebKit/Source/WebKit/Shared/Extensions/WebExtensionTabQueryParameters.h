/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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

#include "WebExtensionWindow.h"
#include "WebExtensionWindowIdentifier.h"
#include <wtf/Forward.h>

namespace WebKit {

struct WebExtensionTabQueryParameters {
    std::optional<Vector<String>> urlPatterns;
    std::optional<String> titlePattern;

    std::optional<WebExtensionWindowIdentifier> windowIdentifier;
    std::optional<OptionSet<WebExtensionWindow::TypeFilter>> windowType;
    std::optional<bool> currentWindow;
    std::optional<bool> frontmostWindow;
    std::optional<size_t> index;

    std::optional<bool> active;
    std::optional<bool> audible;
    std::optional<bool> hidden;
    std::optional<bool> loading;
    std::optional<bool> muted;
    std::optional<bool> pinned;
    std::optional<bool> selected;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
