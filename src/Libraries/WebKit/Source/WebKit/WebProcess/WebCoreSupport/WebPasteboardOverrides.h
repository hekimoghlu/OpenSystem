/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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

#include <wtf/HashMap.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
struct PasteboardItemInfo;
}

namespace WebKit {

class WebPasteboardOverrides {
    friend NeverDestroyed<WebPasteboardOverrides>;
public:
    static WebPasteboardOverrides& sharedPasteboardOverrides();

    void addOverride(const String& pasteboardName, const String& type, const Vector<uint8_t>&);
    void removeOverride(const String& pasteboardName, const String& type);

    std::optional<WebCore::PasteboardItemInfo> overriddenInfo(const String& pasteboardName);
    Vector<String> overriddenTypes(const String& pasteboardName);

    bool getDataForOverride(const String& pasteboardName, const String& type, Vector<uint8_t>&) const;

private:
    WebPasteboardOverrides();

    // The m_overridesMap maps string pasteboard names to pasteboard entries.
    // Each pasteboard entry is a map of a string type to a data buffer.
    HashMap<String, HashMap<String, Vector<uint8_t>>> m_overridesMap;
};

} // namespace WebKit
