/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#include "FontDatabase.h"

#include "FontCacheCoreText.h"
#include "FontCascadeDescription.h"
#include <pal/spi/cf/CoreTextSPI.h>

namespace WebCore {

FontDatabase::InstalledFont::InstalledFont(CTFontDescriptorRef fontDescriptor)
    : fontDescriptor(fontDescriptor)
    , capabilities(capabilitiesForFontDescriptor(fontDescriptor))
{
}

FontDatabase::InstalledFontFamily::InstalledFontFamily(Vector<InstalledFont>&& installedFonts)
    : installedFonts(WTFMove(installedFonts))
{
    for (auto& font : this->installedFonts)
        expand(font);
}

void FontDatabase::InstalledFontFamily::expand(const InstalledFont& installedFont)
{
    capabilities.expand(installedFont.capabilities);
}

const FontDatabase::InstalledFontFamily& FontDatabase::collectionForFamily(const String& familyName)
{
    auto folded = FontCascadeDescription::foldedFamilyName(familyName);
    {
        Locker locker { m_familyNameToFontDescriptorsLock };
        auto it = m_familyNameToFontDescriptors.find(folded);
        if (it != m_familyNameToFontDescriptors.end())
            return *it->value;
    }

    auto installedFontFamily = [&] {
        auto familyNameString = folded.createCFString();
        auto attributes = adoptCF(CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        CFDictionaryAddValue(attributes.get(), kCTFontFamilyNameAttribute, familyNameString.get());
        addAttributesForInstalledFonts(attributes.get(), m_allowUserInstalledFonts);
        auto fontDescriptorToMatch = adoptCF(CTFontDescriptorCreateWithAttributes(attributes.get()));
        auto mandatoryAttributes = installedFontMandatoryAttributes(m_allowUserInstalledFonts);
        if (auto matches = adoptCF(CTFontDescriptorCreateMatchingFontDescriptors(fontDescriptorToMatch.get(), mandatoryAttributes.get()))) {
            auto count = CFArrayGetCount(matches.get());
            Vector<InstalledFont> result(count, [&](size_t i) {
                return InstalledFont(static_cast<CTFontDescriptorRef>(CFArrayGetValueAtIndex(matches.get(), i)));
            });
            return makeUnique<InstalledFontFamily>(WTFMove(result));
        }
        return makeUnique<InstalledFontFamily>();
    }();

    Locker locker { m_familyNameToFontDescriptorsLock };
    return *m_familyNameToFontDescriptors.add(folded.isolatedCopy(), WTFMove(installedFontFamily)).iterator->value;
}

const FontDatabase::InstalledFont& FontDatabase::fontForPostScriptName(const AtomString& postScriptName)
{
    const auto& folded = FontCascadeDescription::foldedFamilyName(postScriptName);
    return m_postScriptNameToFontDescriptors.ensure(folded, [&] {
        auto postScriptNameString = folded.createCFString();
        CFStringRef nameAttribute = kCTFontPostScriptNameAttribute;
        auto attributes = adoptCF(CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        CFDictionaryAddValue(attributes.get(), kCTFontEnabledAttribute, kCFBooleanTrue);
        CFDictionaryAddValue(attributes.get(), nameAttribute, postScriptNameString.get());
        addAttributesForInstalledFonts(attributes.get(), m_allowUserInstalledFonts);
        auto fontDescriptorToMatch = adoptCF(CTFontDescriptorCreateWithAttributes(attributes.get()));
        auto mandatoryAttributes = installedFontMandatoryAttributes(m_allowUserInstalledFonts);
        auto match = adoptCF(CTFontDescriptorCreateMatchingFontDescriptor(fontDescriptorToMatch.get(), mandatoryAttributes.get()));
        return InstalledFont(match.get());
    }).iterator->value;
}

void FontDatabase::clear()
{
    {
        Locker locker { m_familyNameToFontDescriptorsLock };
        m_familyNameToFontDescriptors.clear();
    }
    m_postScriptNameToFontDescriptors.clear();
}

FontDatabase::FontDatabase(AllowUserInstalledFonts allowUserInstalledFonts)
    : m_allowUserInstalledFonts(allowUserInstalledFonts)
{
}

} // namespace WebCore
