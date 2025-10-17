/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#if ENABLE(VIDEO)

#include "CSSPropertyNames.h"
#include "CaptionPreferencesDelegate.h"
#include "CaptionUserPreferences.h"
#include "Color.h"
#include <wtf/TZoneMalloc.h>

#if PLATFORM(COCOA)
OBJC_CLASS WebCaptionUserPreferencesMediaAFWeakObserver;
#endif

namespace WebCore {

class CaptionUserPreferencesMediaAF : public CaptionUserPreferences {
    WTF_MAKE_TZONE_ALLOCATED(CaptionUserPreferencesMediaAF);
public:
    static Ref<CaptionUserPreferencesMediaAF> create(PageGroup&);
    virtual ~CaptionUserPreferencesMediaAF();

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
    CaptionDisplayMode captionDisplayMode() const override;
    void setCaptionDisplayMode(CaptionDisplayMode) override;

    WEBCORE_EXPORT static CaptionDisplayMode platformCaptionDisplayMode();
    WEBCORE_EXPORT static void platformSetCaptionDisplayMode(CaptionDisplayMode);
    WEBCORE_EXPORT static void setCachedCaptionDisplayMode(CaptionDisplayMode);

    bool userPrefersCaptions() const override;
    bool userPrefersSubtitles() const override;
    bool userPrefersTextDescriptions() const final;

    float captionFontSizeScaleAndImportance(bool&) const override;
    bool captionStrokeWidthForFont(float fontSize, const String& language, float& strokeWidth, bool& important) const override;

    void setInterestedInCaptionPreferenceChanges() override;

    void setPreferredLanguage(const String&) override;
    Vector<String> preferredLanguages() const override;

    WEBCORE_EXPORT static Vector<String> platformPreferredLanguages();
    WEBCORE_EXPORT static void platformSetPreferredLanguage(const String&);
    WEBCORE_EXPORT static void setCachedPreferredLanguages(const Vector<String>&);

    void setPreferredAudioCharacteristic(const String&) override;
    Vector<String> preferredAudioCharacteristics() const override;

    void captionPreferencesChanged() override;

    bool shouldFilterTrackMenu() const { return true; }
    
    WEBCORE_EXPORT static void setCaptionPreferencesDelegate(std::unique_ptr<CaptionPreferencesDelegate>&&);
#else
    bool shouldFilterTrackMenu() const { return false; }
#endif

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK) && PLATFORM(COCOA)
    static RefPtr<CaptionUserPreferencesMediaAF> extractCaptionUserPreferencesMediaAF(void* observer);
#endif

    String captionsStyleSheetOverride() const override;
    Vector<RefPtr<AudioTrack>> sortedTrackListForMenu(AudioTrackList*) override;
    Vector<RefPtr<TextTrack>> sortedTrackListForMenu(TextTrackList*, UncheckedKeyHashSet<TextTrack::Kind>) override;
    String displayNameForTrack(AudioTrack*) const override;
    String displayNameForTrack(TextTrack*) const override;

private:
    CaptionUserPreferencesMediaAF(PageGroup&);

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
    void updateTimerFired();

    String captionsWindowCSS() const;
    String captionsBackgroundCSS() const;
    String captionsTextColorCSS() const;
    Color captionsTextColor(bool&) const;
    String captionsDefaultFontCSS() const;
    String windowRoundedCornerRadiusCSS() const;
    String captionsTextEdgeCSS() const;
    String colorPropertyCSS(CSSPropertyID, const Color&, bool) const;
#endif

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK) && PLATFORM(COCOA)
    static RetainPtr<WebCaptionUserPreferencesMediaAFWeakObserver> createWeakObserver(CaptionUserPreferencesMediaAF*);

    RetainPtr<WebCaptionUserPreferencesMediaAFWeakObserver> m_observer;
#endif

#if HAVE(MEDIA_ACCESSIBILITY_FRAMEWORK)
    Timer m_updateStyleSheetTimer;
    bool m_listeningForPreferenceChanges { false };
    bool m_registeringForNotification { false };
#endif
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
