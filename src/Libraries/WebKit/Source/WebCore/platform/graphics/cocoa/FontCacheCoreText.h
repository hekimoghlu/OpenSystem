/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#include "ShouldLocalizeAxisNames.h"

#include <CoreText/CTFont.h>

namespace WebCore {

class FontCreationContext;
class UnrealizedCoreTextFont;
enum class FontLookupOptions : uint8_t;

struct SynthesisPair {
    explicit SynthesisPair(bool needsSyntheticBold, bool needsSyntheticOblique)
        : needsSyntheticBold(needsSyntheticBold)
        , needsSyntheticOblique(needsSyntheticOblique)
    {
    }

    std::pair<bool, bool> boldObliquePair() const
    {
        return std::make_pair(needsSyntheticBold, needsSyntheticOblique);
    }

    bool needsSyntheticBold;
    bool needsSyntheticOblique;
};

struct VariationDefaults {
    String axisName;
    float defaultValue;
    float minimumValue;
    float maximumValue;

    bool contains(float value) const
    {
        ASSERT(minimumValue <= maximumValue);
        return value >= minimumValue && value <= maximumValue;
    }

    float clamp(float value) const
    {
        ASSERT(minimumValue <= maximumValue);
        return std::clamp(value, minimumValue, maximumValue);
    }
};

typedef UncheckedKeyHashMap<FontTag, VariationDefaults, FourCharacterTagHash, FourCharacterTagHashTraits> VariationDefaultsMap;

enum class FontTypeForPreparation : bool {
    SystemFont,
    NonSystemFont
};
enum class ApplyTraitsVariations : bool { No, Yes };
RetainPtr<CTFontRef> preparePlatformFont(UnrealizedCoreTextFont&&, const FontDescription&, const FontCreationContext&, FontTypeForPreparation = FontTypeForPreparation::NonSystemFont, ApplyTraitsVariations = ApplyTraitsVariations::Yes);
enum class ShouldComputePhysicalTraits : bool { No, Yes };
SynthesisPair computeNecessarySynthesis(CTFontRef, const FontDescription&, OptionSet<FontLookupOptions> = { }, ShouldComputePhysicalTraits = ShouldComputePhysicalTraits::No, bool isPlatformFont = false);
RetainPtr<CTFontRef> platformFontWithFamily(const AtomString& family, FontSelectionRequest, TextRenderingMode, float size, OptionSet<FontLookupOptions>);
FontSelectionCapabilities capabilitiesForFontDescriptor(CTFontDescriptorRef);
void addAttributesForInstalledFonts(CFMutableDictionaryRef attributes, AllowUserInstalledFonts);
RetainPtr<CTFontRef> createFontForInstalledFonts(CTFontDescriptorRef, CGFloat size, AllowUserInstalledFonts);
RetainPtr<CTFontRef> createFontForInstalledFonts(CTFontRef, AllowUserInstalledFonts);
void addAttributesForWebFonts(CFMutableDictionaryRef attributes, AllowUserInstalledFonts);
RetainPtr<CFSetRef> installedFontMandatoryAttributes(AllowUserInstalledFonts);
WEBCORE_EXPORT void setOverrideEnhanceTextLegibility(bool);
bool fontNameIsSystemFont(CFStringRef);

CFStringRef getUIContentSizeCategoryDidChangeNotificationName();
WEBCORE_EXPORT void setContentSizeCategory(const String&);
WEBCORE_EXPORT CFStringRef contentSizeCategory();

VariationDefaultsMap defaultVariationValues(CTFontRef, ShouldLocalizeAxisNames);

} // namespace WebCore
