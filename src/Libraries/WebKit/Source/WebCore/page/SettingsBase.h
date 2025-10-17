/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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

#include "ClipboardAccessPolicy.h"
#include "ContentType.h"
#include "EditableLinkBehavior.h"
#include "EditingBehaviorType.h"
#include "FontGenericFamilies.h"
#include "FontLoadTimingOverride.h"
#include "ForcedAccessibilityValue.h"
#include "FourCC.h"
#include "HTMLParserScriptingFlagPolicy.h"
#include "MediaPlayerEnums.h"
#include "StorageBlockingPolicy.h"
#include "StorageMap.h"
#include "TextDirectionSubmenuInclusionBehavior.h"
#include "Timer.h"
#include "TrustedFonts.h"
#include "UserInterfaceDirectionPolicy.h"
#include "WritingMode.h"
#include <JavaScriptCore/RuntimeFlags.h>
#include <unicode/uscript.h>
#include <wtf/AbstractRefCounted.h>
#include <wtf/RefCounted.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>

#if ENABLE(DATA_DETECTION)
#include "DataDetectorType.h"
#endif

namespace WebCore {

class Page;

class SettingsBase : public AbstractRefCounted {
    WTF_MAKE_TZONE_ALLOCATED(SettingsBase);
    WTF_MAKE_NONCOPYABLE(SettingsBase);
public:

#if ENABLE(MEDIA_SOURCE)
    WEBCORE_EXPORT static bool platformDefaultMediaSourceEnabled();
    WEBCORE_EXPORT static uint64_t defaultMaximumSourceBufferSize();
#endif

    static const unsigned defaultMaximumHTMLParserDOMTreeDepth = 512;
    static const unsigned defaultMaximumRenderTreeDepth = 512;

    virtual FontGenericFamilies& fontGenericFamilies() = 0;
    virtual const FontGenericFamilies& fontGenericFamilies() const = 0;

    WEBCORE_EXPORT void setStandardFontFamily(const String&, UScriptCode = USCRIPT_COMMON);
    WEBCORE_EXPORT const String& standardFontFamily(UScriptCode = USCRIPT_COMMON) const;

    WEBCORE_EXPORT void setFixedFontFamily(const String&, UScriptCode = USCRIPT_COMMON);
    WEBCORE_EXPORT const String& fixedFontFamily(UScriptCode = USCRIPT_COMMON) const;

    WEBCORE_EXPORT void setSerifFontFamily(const String&, UScriptCode = USCRIPT_COMMON);
    WEBCORE_EXPORT const String& serifFontFamily(UScriptCode = USCRIPT_COMMON) const;

    WEBCORE_EXPORT void setSansSerifFontFamily(const String&, UScriptCode = USCRIPT_COMMON);
    WEBCORE_EXPORT const String& sansSerifFontFamily(UScriptCode = USCRIPT_COMMON) const;

    WEBCORE_EXPORT void setCursiveFontFamily(const String&, UScriptCode = USCRIPT_COMMON);
    WEBCORE_EXPORT const String& cursiveFontFamily(UScriptCode = USCRIPT_COMMON) const;

    WEBCORE_EXPORT void setFantasyFontFamily(const String&, UScriptCode = USCRIPT_COMMON);
    WEBCORE_EXPORT const String& fantasyFontFamily(UScriptCode = USCRIPT_COMMON) const;

    WEBCORE_EXPORT void setPictographFontFamily(const String&, UScriptCode = USCRIPT_COMMON);
    WEBCORE_EXPORT const String& pictographFontFamily(UScriptCode = USCRIPT_COMMON) const;

    WEBCORE_EXPORT void setMinimumDOMTimerInterval(Seconds); // Initialized to DOMTimer::defaultMinimumInterval().
    Seconds minimumDOMTimerInterval() const { return m_minimumDOMTimerInterval; }

#if ENABLE(TEXT_AUTOSIZING)
    float oneLineTextMultiplierCoefficient() const { return m_oneLineTextMultiplierCoefficient; }
    float multiLineTextMultiplierCoefficient() const { return m_multiLineTextMultiplierCoefficient; }
    float maxTextAutosizingScaleIncrease() const { return m_maxTextAutosizingScaleIncrease; }
#endif

    WEBCORE_EXPORT void setMediaContentTypesRequiringHardwareSupport(const Vector<ContentType>&);
    WEBCORE_EXPORT void setMediaContentTypesRequiringHardwareSupport(const String&);
    const Vector<ContentType>& mediaContentTypesRequiringHardwareSupport() const { return m_mediaContentTypesRequiringHardwareSupport; }

    void setAllowedMediaContainerTypes(std::optional<Vector<String>>&& types) { m_allowedMediaContainerTypes = WTFMove(types); }
    WEBCORE_EXPORT void setAllowedMediaContainerTypes(const String&);
    const std::optional<Vector<String>>& allowedMediaContainerTypes() const { return m_allowedMediaContainerTypes; }

    void setAllowedMediaCodecTypes(std::optional<Vector<String>>&& types) { m_allowedMediaCodecTypes = WTFMove(types); }
    WEBCORE_EXPORT void setAllowedMediaCodecTypes(const String&);
    const std::optional<Vector<String>>& allowedMediaCodecTypes() const { return m_allowedMediaCodecTypes; }

    void setAllowedMediaVideoCodecIDs(std::optional<Vector<FourCC>>&& types) { m_allowedMediaVideoCodecIDs = WTFMove(types); }
    WEBCORE_EXPORT void setAllowedMediaVideoCodecIDs(const String&);
    const std::optional<Vector<FourCC>>& allowedMediaVideoCodecIDs() const { return m_allowedMediaVideoCodecIDs; }

    void setAllowedMediaAudioCodecIDs(std::optional<Vector<FourCC>>&& types) { m_allowedMediaAudioCodecIDs = WTFMove(types); }
    WEBCORE_EXPORT void setAllowedMediaAudioCodecIDs(const String&);
    const std::optional<Vector<FourCC>>& allowedMediaAudioCodecIDs() const { return m_allowedMediaAudioCodecIDs; }

    void setAllowedMediaCaptionFormatTypes(std::optional<Vector<FourCC>>&& types) { m_allowedMediaCaptionFormatTypes = WTFMove(types); }
    WEBCORE_EXPORT void setAllowedMediaCaptionFormatTypes(const String&);
    const std::optional<Vector<FourCC>>& allowedMediaCaptionFormatTypes() const { return m_allowedMediaCaptionFormatTypes; }

    WEBCORE_EXPORT void resetToConsistentState();

protected:
    explicit SettingsBase(Page*);
    virtual ~SettingsBase();

    void initializeDefaultFontFamilies();

    void imageLoadingSettingsTimerFired();

    // Helpers used by generated Settings.cpp.
    void setNeedsRecalcStyleInAllFrames();
    void setNeedsRelayoutAllFrames();
    void mediaTypeOverrideChanged();
    void imagesEnabledChanged();
    void userStyleSheetLocationChanged();
    void usesBackForwardCacheChanged();
    void dnsPrefetchingEnabledChanged();
    void storageBlockingPolicyChanged();
    void backgroundShouldExtendBeyondPageChanged();
    void scrollingPerformanceTestingEnabledChanged();
    void hiddenPageDOMTimerThrottlingStateChanged();
    void hiddenPageCSSAnimationSuspensionEnabledChanged();
    void resourceUsageOverlayVisibleChanged();
    void iceCandidateFilteringEnabledChanged();
#if ENABLE(TEXT_AUTOSIZING)
    void shouldEnableTextAutosizingBoostChanged();
    void textAutosizingUsesIdempotentModeChanged();
#endif
#if ENABLE(MEDIA_STREAM)
    void mockCaptureDevicesEnabledChanged();
#endif
    void layerBasedSVGEngineEnabledChanged();
#if USE(MODERN_AVCONTENTKEYSESSION)
    void shouldUseModernAVContentKeySessionChanged();
#endif
    void useSystemAppearanceChanged();
    void fontFallbackPrefersPictographsChanged();
    RefPtr<Page> protectedPage() const;

    WeakPtr<Page> m_page;

    Seconds m_minimumDOMTimerInterval;

    Timer m_setImageLoadingSettingsTimer;

    Vector<ContentType> m_mediaContentTypesRequiringHardwareSupport;
    std::optional<Vector<String>> m_allowedMediaContainerTypes;
    std::optional<Vector<String>> m_allowedMediaCodecTypes;
    std::optional<Vector<FourCC>> m_allowedMediaVideoCodecIDs;
    std::optional<Vector<FourCC>> m_allowedMediaAudioCodecIDs;
    std::optional<Vector<FourCC>> m_allowedMediaCaptionFormatTypes;

#if ENABLE(TEXT_AUTOSIZING)
    static constexpr const float boostedOneLineTextMultiplierCoefficient = 2.23125f;
    static constexpr const float boostedMultiLineTextMultiplierCoefficient = 2.48125f;
    static constexpr const float boostedMaxTextAutosizingScaleIncrease = 5.0f;
    static constexpr const float defaultOneLineTextMultiplierCoefficient = 1.7f;
    static constexpr const float defaultMultiLineTextMultiplierCoefficient = 1.95f;
    static constexpr const float defaultMaxTextAutosizingScaleIncrease = 1.7f;

    float m_oneLineTextMultiplierCoefficient { defaultOneLineTextMultiplierCoefficient };
    float m_multiLineTextMultiplierCoefficient { defaultMultiLineTextMultiplierCoefficient };
    float m_maxTextAutosizingScaleIncrease { defaultMaxTextAutosizingScaleIncrease };
#endif
};

} // namespace WebCore
