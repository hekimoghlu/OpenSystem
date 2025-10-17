/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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

#if USE(CAIRO) || USE(SKIA)
#include <optional>
#include <wtf/NeverDestroyed.h>

#if USE(CAIRO)
#include "CairoUniquePtr.h"
#elif USE(SKIA)
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkFont.h>
#include <skia/core/SkFontTypes.h>
#include <skia/core/SkSurfaceProps.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#endif

#if PLATFORM(GTK) && !USE(GTK4)
static constexpr bool s_followSystemSettingsDefault = true;
#else
static constexpr bool s_followSystemSettingsDefault = false;
#endif

namespace WebCore {

class FontRenderOptions {
    friend class NeverDestroyed<FontRenderOptions>;
public:
    WEBCORE_EXPORT static FontRenderOptions& singleton();

    enum class Hinting {
        None,
        Slight,
        Medium,
        Full
    };

    enum class Antialias {
        None,
        Normal,
        Subpixel
    };

    enum class SubpixelOrder {
        Unknown,
        HorizontalRGB,
        HorizontalBGR,
        VerticalRGB,
        VerticalBGR
    };

    void setHinting(std::optional<Hinting>);
    void setAntialias(std::optional<Antialias>);
    void setSubpixelOrder(std::optional<SubpixelOrder>);
    void setFollowSystemSettings(std::optional<bool> followSystemSettings) { m_followSystemSettings = followSystemSettings.value_or(s_followSystemSettingsDefault); }

#if USE(CAIRO)
    const cairo_font_options_t* fontOptions() const { return m_fontOptions.get(); }
#elif USE(SKIA)
    SkFontHinting hinting() const;
    SkFont::Edging antialias() const;
    SkPixelGeometry subpixelOrder() const;
    void setUseSubpixelPositioning(bool enable) { m_useSubpixelPositioning = enable; }
    bool useSubpixelPositioning() const;
#endif

    WEBCORE_EXPORT void disableHintingForTesting();
    bool isHintingDisabledForTesting() const { return m_isHintingDisabledForTesting; }

private:
    FontRenderOptions();
    ~FontRenderOptions() = default;

#if USE(CAIRO)
    CairoUniquePtr<cairo_font_options_t> m_fontOptions;
#elif USE(SKIA)
    SkFontHinting m_hinting { SkFontHinting::kNormal };
    SkFont::Edging m_antialias { SkFont::Edging::kAntiAlias };
    SkPixelGeometry m_subpixelOrder { kUnknown_SkPixelGeometry };
    bool m_useSubpixelPositioning { false };
#endif
    bool m_followSystemSettings { s_followSystemSettingsDefault };
    bool m_isHintingDisabledForTesting { false };
};

} // namespace WebCore

#endif // USE(CAIRO) || USE(SKIA)
