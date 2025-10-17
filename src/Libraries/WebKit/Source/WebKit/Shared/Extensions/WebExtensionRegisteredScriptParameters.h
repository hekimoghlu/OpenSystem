/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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

#include <WebCore/UserStyleSheetTypes.h>
#include <wtf/Forward.h>

namespace WebKit {

struct WebExtensionRegisteredScriptParameters {

    std::optional<Vector<String>> css;
    std::optional<Vector<String>> js;

    String identifier;
    std::optional<WebExtension::InjectionTime> injectionTime;

    std::optional<Vector<String>> excludeMatchPatterns;
    std::optional<Vector<String>> matchPatterns;

    std::optional<bool> allFrames;
    std::optional<WebCore::UserContentMatchParentFrame> matchParentFrame;

    std::optional<bool> persistent;

    std::optional<WebExtensionContentWorldType> world;
    std::optional<WebCore::UserStyleLevel> styleLevel;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
