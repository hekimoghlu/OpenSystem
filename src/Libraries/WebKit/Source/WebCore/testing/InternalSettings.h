/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

#include "ExceptionOr.h"
#include "FontGenericFamilies.h"
#include "InternalSettingsGenerated.h"
#include "Settings.h"

namespace WebCore {

class Page;

class InternalSettings : public InternalSettingsGenerated {
public:
    static Ref<InternalSettings> create(Page*);
    static InternalSettings* from(Page*);
    void hostDestroyed();
    void resetToConsistentState();

    // Settings
    ExceptionOr<void> setStandardFontFamily(const String& family, const String& script);
    ExceptionOr<void> setSerifFontFamily(const String& family, const String& script);
    ExceptionOr<void> setSansSerifFontFamily(const String& family, const String& script);
    ExceptionOr<void> setFixedFontFamily(const String& family, const String& script);
    ExceptionOr<void> setCursiveFontFamily(const String& family, const String& script);
    ExceptionOr<void> setFantasyFontFamily(const String& family, const String& script);
    ExceptionOr<void> setPictographFontFamily(const String& family, const String& script);

    ExceptionOr<void> setTextAutosizingWindowSizeOverride(int width, int height);

    ExceptionOr<void> setMinimumTimerInterval(double intervalInSeconds);
    ExceptionOr<void> setTimeWithoutMouseMovementBeforeHidingControls(double intervalInSeconds);

    using EditingBehaviorType = WebCore::EditingBehaviorType;
    ExceptionOr<void> setEditingBehavior(EditingBehaviorType);
    
    using StorageBlockingPolicy = WebCore::StorageBlockingPolicy;
    ExceptionOr<void> setStorageBlockingPolicy(StorageBlockingPolicy);
    
    using UserInterfaceDirectionPolicy = WebCore::UserInterfaceDirectionPolicy;
    ExceptionOr<void> setUserInterfaceDirectionPolicy(UserInterfaceDirectionPolicy);

    using SystemLayoutDirection = TextDirection;
    ExceptionOr<void> setSystemLayoutDirection(SystemLayoutDirection);

    using FontLoadTimingOverride = WebCore::FontLoadTimingOverride;
    ExceptionOr<void> setFontLoadTimingOverride(FontLoadTimingOverride);

    using ForcedAccessibilityValue = WebCore::ForcedAccessibilityValue;
    ForcedAccessibilityValue forcedColorsAreInvertedAccessibilityValue() const;
    void setForcedColorsAreInvertedAccessibilityValue(ForcedAccessibilityValue);
    ForcedAccessibilityValue forcedDisplayIsMonochromeAccessibilityValue() const;
    void setForcedDisplayIsMonochromeAccessibilityValue(ForcedAccessibilityValue);
    ForcedAccessibilityValue forcedPrefersContrastAccessibilityValue() const;
    void setForcedPrefersContrastAccessibilityValue(ForcedAccessibilityValue);
    ForcedAccessibilityValue forcedPrefersReducedMotionAccessibilityValue() const;
    void setForcedPrefersReducedMotionAccessibilityValue(ForcedAccessibilityValue);
    ForcedAccessibilityValue forcedSupportsHighDynamicRangeValue() const;
    void setForcedSupportsHighDynamicRangeValue(ForcedAccessibilityValue);

    ExceptionOr<void> setAllowAnimationControlsOverride(bool);

    // DeprecatedGlobalSettings.
    ExceptionOr<void> setCustomPasteboardDataEnabled(bool);

    bool vp9DecoderEnabled() const;

    ExceptionOr<void> setShouldManageAudioSessionCategory(bool);

    // CaptionUserPreferences.
    enum class TrackKind : uint8_t { Subtitles, Captions, TextDescriptions };
    ExceptionOr<void> setShouldDisplayTrackKind(TrackKind, bool enabled);
    ExceptionOr<bool> shouldDisplayTrackKind(TrackKind);

    // Page
    ExceptionOr<void> setEditableRegionEnabled(bool);
    ExceptionOr<void> setCanStartMedia(bool);
    ExceptionOr<void> setUseDarkAppearance(bool);
    ExceptionOr<void> setUseElevatedUserInterfaceLevel(bool);

    // ScrollView
    ExceptionOr<void> setAllowUnclampedScrollPosition(bool);

    // PlatformMediaSessionManager.
    ExceptionOr<void> setShouldDeactivateAudioSession(bool);

    // RenderTheme/FontCache
    ExceptionOr<void> setShouldMockBoldSystemFontForAccessibility(bool);

    // AudioContext
    ExceptionOr<void> setDefaultAudioContextSampleRate(float);

    ExceptionOr<void> setAllowedMediaContainerTypes(const String&);
    ExceptionOr<void> setAllowedMediaCodecTypes(const String&);
    ExceptionOr<void> setAllowedMediaVideoCodecIDs(const String&);
    ExceptionOr<void> setAllowedMediaAudioCodecIDs(const String&);
    ExceptionOr<void> setAllowedMediaCaptionFormatTypes(const String&);

private:
    explicit InternalSettings(Page*);

    Settings& settings() const;
    static ASCIILiteral supplementName();

    class Backup {
    public:
        explicit Backup(Settings&);
        void restoreTo(Settings&);

        // Settings
        // ScriptFontFamilyMaps are initially empty, only used if changed by a test.
        ScriptFontFamilyMap m_standardFontFamilies;
        ScriptFontFamilyMap m_fixedFontFamilies;
        ScriptFontFamilyMap m_serifFontFamilies;
        ScriptFontFamilyMap m_sansSerifFontFamilies;
        ScriptFontFamilyMap m_cursiveFontFamilies;
        ScriptFontFamilyMap m_fantasyFontFamilies;
        ScriptFontFamilyMap m_pictographFontFamilies;
        Seconds m_minimumDOMTimerInterval;
        Seconds m_originalTimeWithoutMouseMovementBeforeHidingControls;
        WebCore::EditingBehaviorType m_originalEditingBehavior;
        WebCore::StorageBlockingPolicy m_storageBlockingPolicy;
        WebCore::UserInterfaceDirectionPolicy m_userInterfaceDirectionPolicy;
        TextDirection m_systemLayoutDirection;
        WebCore::ForcedAccessibilityValue m_forcedColorsAreInvertedAccessibilityValue;
        WebCore::ForcedAccessibilityValue m_forcedDisplayIsMonochromeAccessibilityValue;
        WebCore::ForcedAccessibilityValue m_forcedPrefersContrastAccessibilityValue;
        WebCore::ForcedAccessibilityValue m_forcedPrefersReducedMotionAccessibilityValue;
        WebCore::FontLoadTimingOverride m_fontLoadTimingOverride;

        // DeprecatedGlobalSettings
        bool m_customPasteboardDataEnabled;
        bool m_originalMockScrollbarsEnabled;
#if USE(AUDIO_SESSION)
        bool m_shouldManageAudioSessionCategory;
#endif

        // PlatformMediaSessionManager
        bool m_shouldDeactivateAudioSession;
    };

    WeakPtr<Page> m_page;
    Backup m_backup;
};

} // namespace WebCore
