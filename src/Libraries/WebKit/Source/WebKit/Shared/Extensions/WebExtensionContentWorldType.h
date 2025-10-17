/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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

#include <wtf/text/WTFString.h>

namespace WebKit {

enum class WebExtensionContentWorldType : uint8_t {
    Main,
    ContentScript,
    Native,
    WebPage,
#if ENABLE(INSPECTOR_EXTENSIONS)
    Inspector,
#endif
};

inline bool isEqual(WebExtensionContentWorldType a, WebExtensionContentWorldType b)
{
#if ENABLE(INSPECTOR_EXTENSIONS)
    // Inspector content world is a special alias of Main. Consider them equal.
    if ((a == WebExtensionContentWorldType::Main || a == WebExtensionContentWorldType::Inspector) && (b == WebExtensionContentWorldType::Main || b == WebExtensionContentWorldType::Inspector))
        return true;
#endif
    return a == b;
}

inline String toDebugString(WebExtensionContentWorldType contentWorldType)
{
    switch (contentWorldType) {
    case WebExtensionContentWorldType::Main:
        return "main"_s;
    case WebExtensionContentWorldType::ContentScript:
        return "content script"_s;
    case WebExtensionContentWorldType::Native:
        return "native"_s;
    case WebExtensionContentWorldType::WebPage:
        return "web page"_s;
#if ENABLE(INSPECTOR_EXTENSIONS)
    case WebExtensionContentWorldType::Inspector:
        return "inspector"_s;
#endif
    }
}

} // namespace WebKit

namespace WTF {

template<> struct DefaultHash<WebKit::WebExtensionContentWorldType> : IntHash<WebKit::WebExtensionContentWorldType> { };
template<> struct HashTraits<WebKit::WebExtensionContentWorldType> : StrongEnumHashTraits<WebKit::WebExtensionContentWorldType> { };

} // namespace WTF

#endif // ENABLE(WK_WEB_EXTENSIONS)
