/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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

#include "FontFeatureValues.h"
#include "FontPaletteValues.h"
#include "FontSelectionAlgorithm.h"
#include "FontTaggedSettings.h"
#include <wtf/PointerComparison.h>

namespace WebCore {

class FontCreationContextRareData : public RefCounted<FontCreationContextRareData> {
public:
    static Ref<FontCreationContextRareData> create(const FontFeatureSettings& fontFaceFeatures, const FontPaletteValues& fontPaletteValues, RefPtr<FontFeatureValues> fontFeatureValues, float sizeAdjust)
    {
        return adoptRef(*new FontCreationContextRareData(fontFaceFeatures, fontPaletteValues, fontFeatureValues, sizeAdjust));
    }

    const FontFeatureSettings& fontFaceFeatures() const
    {
        return m_fontFaceFeatures;
    }

    float sizeAdjust() const
    {
        return m_sizeAdjust;
    }

    const FontPaletteValues& fontPaletteValues() const
    {
        return m_fontPaletteValues;
    }

    RefPtr<FontFeatureValues> fontFeatureValues() const
    {
        return m_fontFeatureValues;
    }

    bool operator==(const FontCreationContextRareData& other) const
    {
        return m_fontFaceFeatures == other.m_fontFaceFeatures
            && m_fontPaletteValues == other.m_fontPaletteValues
            && m_fontFeatureValues.get() == other.m_fontFeatureValues.get()
            && m_sizeAdjust == other.m_sizeAdjust;
    }

private:
    FontCreationContextRareData(const FontFeatureSettings& fontFaceFeatures, const FontPaletteValues& fontPaletteValues, RefPtr<FontFeatureValues> fontFeatureValues, float sizeAdjust)
        : m_fontFaceFeatures(fontFaceFeatures)
        , m_fontPaletteValues(fontPaletteValues)
        , m_fontFeatureValues(fontFeatureValues)
        , m_sizeAdjust(sizeAdjust)
    {
    }

    FontFeatureSettings m_fontFaceFeatures;
    FontPaletteValues m_fontPaletteValues;
    RefPtr<FontFeatureValues> m_fontFeatureValues;
    float m_sizeAdjust;
};

class FontCreationContext {
public:
    FontCreationContext() = default;

    FontCreationContext(const FontFeatureSettings& fontFaceFeatures, const FontSelectionSpecifiedCapabilities& fontFaceCapabilities, const FontPaletteValues& fontPaletteValues, RefPtr<FontFeatureValues> fontFeatureValues, float sizeAdjust)
        : m_fontFaceCapabilities(fontFaceCapabilities)
    {
        if (!fontFaceFeatures.isEmpty() || fontPaletteValues || (fontFeatureValues && !fontFeatureValues->isEmpty()) || sizeAdjust != 1.0)
            m_rareData = FontCreationContextRareData::create(fontFaceFeatures, fontPaletteValues, fontFeatureValues, sizeAdjust);
    }

    const FontFeatureSettings* fontFaceFeatures() const
    {
        return m_rareData ? &m_rareData->fontFaceFeatures() : nullptr;
    }

    float sizeAdjust() const
    {
        return m_rareData ? m_rareData->sizeAdjust() : 1.0;
    }

    const FontSelectionSpecifiedCapabilities& fontFaceCapabilities() const
    {
        return m_fontFaceCapabilities;
    }

    const FontPaletteValues* fontPaletteValues() const
    {
        return m_rareData ? &m_rareData->fontPaletteValues() : nullptr;
    }

    RefPtr<FontFeatureValues> fontFeatureValues() const
    {
        return m_rareData ? m_rareData->fontFeatureValues() : nullptr;
    }

    bool operator==(const FontCreationContext& other) const
    {
        return m_fontFaceCapabilities == other.m_fontFaceCapabilities
            && arePointingToEqualData(m_rareData, other.m_rareData);
    }

private:
    FontSelectionSpecifiedCapabilities m_fontFaceCapabilities;
    RefPtr<FontCreationContextRareData> m_rareData;
};

inline void add(Hasher& hasher, const FontCreationContext& fontCreationContext)
{
    if (fontCreationContext.fontFaceFeatures())
        add(hasher, *fontCreationContext.fontFaceFeatures());
    add(hasher, fontCreationContext.fontFaceCapabilities());
    if (fontCreationContext.fontPaletteValues())
        add(hasher, *fontCreationContext.fontPaletteValues());
    if (fontCreationContext.fontFeatureValues())
        add(hasher, *fontCreationContext.fontFeatureValues());
    if (fontCreationContext.sizeAdjust())
        add(hasher, fontCreationContext.sizeAdjust());
}

}
