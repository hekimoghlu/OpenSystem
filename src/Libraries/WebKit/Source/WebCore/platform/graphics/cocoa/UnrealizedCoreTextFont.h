/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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

#include "FontCacheCoreText.h"
#include <CoreFoundation/CoreFoundation.h>
#include <CoreText/CoreText.h>
#include <optional>
#include <variant>
#include <wtf/RetainPtr.h>

namespace WebCore {

class FontCreationContext;
class FontDescription;

class UnrealizedCoreTextFont {
public:
    UnrealizedCoreTextFont(RetainPtr<CTFontRef>&& baseFont)
        : m_baseFont(WTFMove(baseFont))
    {
    }

    UnrealizedCoreTextFont(RetainPtr<CTFontDescriptorRef>&& baseFont)
        : m_baseFont(WTFMove(baseFont))
    {
    }

    template <typename T>
    void modify(T&& functor)
    {
        if (static_cast<bool>(*this))
            functor(m_attributes.get());
    }

    void setSize(CGFloat size)
    {
        if (static_cast<bool>(*this))
            CFDictionarySetValue(m_attributes.get(), kCTFontSizeAttribute, adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberCGFloatType, &size)).get());
    }

    operator bool() const;

    void modifyFromContext(const FontDescription&, const FontCreationContext&, FontTypeForPreparation = FontTypeForPreparation::NonSystemFont, ApplyTraitsVariations = ApplyTraitsVariations::Yes, bool shouldEnhanceTextLegibility = false);

    RetainPtr<CTFontRef> realize() const;

private:
    CGFloat getSize() const;

    struct OpticalSizingTypes { // Ideally this would be a namespace, but clang doesn't seem to let you define a namespace inside a class.
        // When USE(CORE_TEXT_OPTICAL_SIZING_WORKAROUND) is no longer necessary, we can migrate this back to an enum.
        struct None { };
        struct JustVariation { };
        struct Everything {
            std::optional<float> opticalSizingValue;
        };
    };

    using OpticalSizingType = std::variant<OpticalSizingTypes::None, OpticalSizingTypes::JustVariation, OpticalSizingTypes::Everything>;

    static void modifyFromContext(CFMutableDictionaryRef attributes, const FontDescription&, const FontCreationContext&, ApplyTraitsVariations, float weight, float width, float slope, CGFloat size, const OpticalSizingType&);

    using VariationsMap = UncheckedKeyHashMap<FontTag, float, FourCharacterTagHash, FourCharacterTagHashTraits>;
    static void addAttributesForOpticalSizing(CFMutableDictionaryRef attributes, VariationsMap& variationsToBeApplied, const OpticalSizingType&, CGFloat size);
    static void applyVariations(CFMutableDictionaryRef attributes, const VariationsMap& variationsToBeApplied);

    struct RebuildReason {
        bool gxVariations { false };
        std::optional<VariationDefaultsMap> variationDefaults;

        bool hasEffect() const
        {
            return gxVariations || variationDefaults;
        }
    };
    RebuildReason rebuildReason(CTFontRef) const;

    std::variant<RetainPtr<CTFontRef>, RetainPtr<CTFontDescriptorRef>> m_baseFont;
    RetainPtr<CFMutableDictionaryRef> m_attributes { adoptCF(CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks)) };

    ApplyTraitsVariations m_applyTraitsVariations { ApplyTraitsVariations::Yes };
    float m_weight { 0 };
    float m_width { 0 };
    float m_slope { 0 };
    CGFloat m_size { 0 };
    FontStyleAxis m_fontStyleAxis { FontStyleAxis::slnt };
    OpticalSizingType m_opticalSizingType { OpticalSizingTypes::None { } };
    FontVariationSettings m_variationSettings;
};

} // namespace WebCore
