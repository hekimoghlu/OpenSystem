/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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

#include "FontSelectionAlgorithm.h"
#include "TextFlags.h"
#include <CoreText/CoreText.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class FontDatabase {
    WTF_MAKE_NONCOPYABLE(FontDatabase);

public:
    FontDatabase(AllowUserInstalledFonts);

    struct InstalledFont {
        InstalledFont() = default;
        InstalledFont(CTFontDescriptorRef);

        RetainPtr<CTFontDescriptorRef> fontDescriptor;
        FontSelectionCapabilities capabilities;
    };

    struct InstalledFontFamily {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        InstalledFontFamily() = default;

        explicit InstalledFontFamily(Vector<InstalledFont>&& installedFonts);

        void expand(const InstalledFont&);
        bool isEmpty() const { return installedFonts.isEmpty(); }
        size_t size() const { return installedFonts.size(); }

        Vector<InstalledFont> installedFonts;
        FontSelectionCapabilities capabilities;
    };

    AllowUserInstalledFonts allowUserInstalledFonts() const { return m_allowUserInstalledFonts; }
    const InstalledFontFamily& collectionForFamily(const String& familyName);
    const InstalledFont& fontForPostScriptName(const AtomString& postScriptName);
    void clear();

private:
    Lock m_familyNameToFontDescriptorsLock;
    MemoryCompactRobinHoodHashMap<String, std::unique_ptr<InstalledFontFamily>> m_familyNameToFontDescriptors WTF_GUARDED_BY_LOCK(m_familyNameToFontDescriptorsLock);
    MemoryCompactRobinHoodHashMap<String, InstalledFont> m_postScriptNameToFontDescriptors;
    AllowUserInstalledFonts m_allowUserInstalledFonts;
};

} // namespace WebCore
