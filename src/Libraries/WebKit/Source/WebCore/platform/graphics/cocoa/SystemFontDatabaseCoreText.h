/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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

#include "FontDescription.h"
#include "SystemFontDatabase.h"
#include <pal/spi/cf/CoreTextSPI.h>
#include <wtf/HashMap.h>
#include <wtf/HashTraits.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

enum class SystemFontKind : uint8_t {
    SystemUI,
    UISerif,
    UIMonospace,
    UIRounded,
    TextStyle
};

class SystemFontDatabaseCoreText : public SystemFontDatabase {
public:
    struct CascadeListParameters {
        CascadeListParameters()
        {
        }

        CascadeListParameters(WTF::HashTableDeletedValueType)
            : fontName(WTF::HashTableDeletedValue)
        {
        }

        bool isHashTableDeletedValue() const
        {
            return fontName.isHashTableDeletedValue();
        }

        friend bool operator==(const CascadeListParameters&, const CascadeListParameters&) = default;

        struct Hash {
            static unsigned hash(const CascadeListParameters&);
            static bool equal(const CascadeListParameters& a, const CascadeListParameters& b) { return a == b; }
            static const bool safeToCompareToEmptyOrDeleted = true;
        };

        AtomString fontName;
        AtomString locale;
        CGFloat weight { 0 };
        CGFloat width { 0 };
        float size { 0 };
        AllowUserInstalledFonts allowUserInstalledFonts { AllowUserInstalledFonts::No };
        bool italic { false };
    };

    SystemFontDatabaseCoreText();
    static SystemFontDatabaseCoreText& forCurrentThread();

    std::optional<SystemFontKind> matchSystemFontUse(const AtomString& family);
    Vector<RetainPtr<CTFontDescriptorRef>> cascadeList(const FontDescription&, const AtomString& cssFamily, SystemFontKind, AllowUserInstalledFonts);

    String serifFamily(const String& locale);
    String sansSerifFamily(const String& locale);
    String cursiveFamily(const String& locale);
    String fantasyFamily(const String& locale);
    String monospaceFamily(const String& locale);

    const AtomString& systemFontShorthandFamily(FontShorthand);
    float systemFontShorthandSize(FontShorthand);
    FontSelectionValue systemFontShorthandWeight(FontShorthand);

    void clear();

private:
    friend class SystemFontDatabase;

    Vector<RetainPtr<CTFontDescriptorRef>> cascadeList(const CascadeListParameters&, SystemFontKind);

    RetainPtr<CTFontRef> createSystemUIFont(const CascadeListParameters&, CFStringRef locale);
    RetainPtr<CTFontRef> createSystemDesignFont(SystemFontKind, const CascadeListParameters&);
    RetainPtr<CTFontRef> createTextStyleFont(const CascadeListParameters&);

    static RetainPtr<CTFontDescriptorRef> smallCaptionFontDescriptor();
    static RetainPtr<CTFontDescriptorRef> menuFontDescriptor();
    static RetainPtr<CTFontDescriptorRef> statusBarFontDescriptor();
    static RetainPtr<CTFontDescriptorRef> miniControlFontDescriptor();
    static RetainPtr<CTFontDescriptorRef> smallControlFontDescriptor();
    static RetainPtr<CTFontDescriptorRef> controlFontDescriptor();

    static RetainPtr<CTFontRef> createFontByApplyingWeightWidthItalicsAndFallbackBehavior(CTFontRef, CGFloat weight, CGFloat width, bool italic, float size, AllowUserInstalledFonts, CFStringRef design = nullptr);
    static RetainPtr<CTFontDescriptorRef> removeCascadeList(CTFontDescriptorRef);
    static Vector<RetainPtr<CTFontDescriptorRef>> computeCascadeList(CTFontRef, CFStringRef locale);
    static CascadeListParameters systemFontParameters(const FontDescription&, const AtomString& familyName, SystemFontKind, AllowUserInstalledFonts);

    UncheckedKeyHashMap<CascadeListParameters, Vector<RetainPtr<CTFontDescriptorRef>>, CascadeListParameters::Hash, SimpleClassHashTraits<CascadeListParameters>> m_systemFontCache;

    MemoryCompactRobinHoodHashMap<String, String> m_serifFamilies;
    MemoryCompactRobinHoodHashMap<String, String> m_sansSeriferifFamilies;
    MemoryCompactRobinHoodHashMap<String, String> m_cursiveFamilies;
    MemoryCompactRobinHoodHashMap<String, String> m_fantasyFamilies;
    MemoryCompactRobinHoodHashMap<String, String> m_monospaceFamilies;

    Vector<AtomString> m_textStyles;
};

inline void add(Hasher& hasher, const SystemFontDatabaseCoreText::CascadeListParameters& parameters)
{
    add(hasher, parameters.fontName, parameters.locale, parameters.weight, parameters.width, parameters.size, parameters.allowUserInstalledFonts, parameters.italic);
}

inline unsigned SystemFontDatabaseCoreText::CascadeListParameters::Hash::hash(const CascadeListParameters& parameters)
{
    return computeHash(parameters);
}

} // namespace WebCore
