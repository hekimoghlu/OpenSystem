/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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

#include "AudioTrack.h"
#include "TextTrack.h"
#include "Timer.h"
#include <wtf/EnumTraits.h>
#include <wtf/HashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CaptionUserPreferencesTestingModeToken;
class HTMLMediaElement;
class Page;
class PageGroup;
class AudioTrackList;
class TextTrackList;
struct MediaSelectionOption;

enum class CaptionUserPreferencesDisplayMode : uint8_t {
    Automatic,
    ForcedOnly,
    AlwaysOn,
    Manual,
};

class CaptionUserPreferences : public RefCountedAndCanMakeWeakPtr<CaptionUserPreferences> {
    WTF_MAKE_TZONE_ALLOCATED(CaptionUserPreferences);
public:
    static Ref<CaptionUserPreferences> create(PageGroup&);
    virtual ~CaptionUserPreferences();

    using CaptionDisplayMode = CaptionUserPreferencesDisplayMode;
    virtual CaptionDisplayMode captionDisplayMode() const;
    virtual void setCaptionDisplayMode(CaptionDisplayMode);

    virtual int textTrackSelectionScore(TextTrack*, HTMLMediaElement*) const;
    virtual int textTrackLanguageSelectionScore(TextTrack*, const Vector<String>&) const;

    virtual bool userPrefersCaptions() const;
    virtual void setUserPrefersCaptions(bool);

    virtual bool userPrefersSubtitles() const;
    virtual void setUserPrefersSubtitles(bool preference);
    
    virtual bool userPrefersTextDescriptions() const;
    virtual void setUserPrefersTextDescriptions(bool preference);

    virtual float captionFontSizeScaleAndImportance(bool& important) const { important = false; return 0.05f; }

    virtual bool captionStrokeWidthForFont(float, const String&, float&, bool&) const { return false; }

    virtual String captionsStyleSheetOverride() const { return m_captionsStyleSheetOverride; }
    virtual void setCaptionsStyleSheetOverride(const String&);

    virtual void setInterestedInCaptionPreferenceChanges() { }

    virtual void captionPreferencesChanged();

    virtual void setPreferredLanguage(const String&);
    virtual Vector<String> preferredLanguages() const;

    virtual void setPreferredAudioCharacteristic(const String&);
    virtual Vector<String> preferredAudioCharacteristics() const;

    virtual String displayNameForTrack(TextTrack*) const;
    MediaSelectionOption mediaSelectionOptionForTrack(TextTrack*) const;
    virtual Vector<RefPtr<TextTrack>> sortedTrackListForMenu(TextTrackList*, UncheckedKeyHashSet<TextTrack::Kind>);

    virtual String displayNameForTrack(AudioTrack*) const;
    MediaSelectionOption mediaSelectionOptionForTrack(AudioTrack*) const;
    virtual Vector<RefPtr<AudioTrack>> sortedTrackListForMenu(AudioTrackList*);

    void setPrimaryAudioTrackLanguageOverride(const String& language) { m_primaryAudioTrackLanguageOverride = language;  }
    String primaryAudioTrackLanguageOverride() const;

    virtual bool testingMode() const { return m_testingModeCount; }

    friend class CaptionUserPreferencesTestingModeToken;
    WEBCORE_EXPORT UniqueRef<CaptionUserPreferencesTestingModeToken> createTestingModeToken();
    
    PageGroup& pageGroup() const;

protected:
    explicit CaptionUserPreferences(PageGroup&);

    void updateCaptionStyleSheetOverride();
    void beginBlockingNotifications();
    void endBlockingNotifications();

private:
    void incrementTestingModeCount() { ++m_testingModeCount; }
    void decrementTestingModeCount()
    {
        ASSERT(m_testingModeCount);
        if (m_testingModeCount)
            --m_testingModeCount;
    }

    void timerFired();
    void notify();
    Page* currentPage() const;

    WeakRef<PageGroup> m_pageGroup;
    mutable CaptionDisplayMode m_displayMode;
    Timer m_timer;
    String m_userPreferredLanguage;
    String m_userPreferredAudioCharacteristic;
    String m_captionsStyleSheetOverride;
    String m_primaryAudioTrackLanguageOverride;
    unsigned m_blockNotificationsCounter { 0 };
    bool m_havePreferences { false };
    unsigned m_testingModeCount { 0 };
};

class CaptionUserPreferencesTestingModeToken {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CaptionUserPreferencesTestingModeToken, WEBCORE_EXPORT);
public:
    CaptionUserPreferencesTestingModeToken(CaptionUserPreferences& parent)
        : m_parent(parent)
    {
        parent.incrementTestingModeCount();
    }
    ~CaptionUserPreferencesTestingModeToken()
    {
        if (m_parent)
            m_parent->decrementTestingModeCount();
    }
private:
    WeakPtr<CaptionUserPreferences> m_parent;
};
    
}

namespace WTF {

template<> struct EnumTraits<WebCore::CaptionUserPreferences::CaptionDisplayMode> {
    static std::optional<WebCore::CaptionUserPreferences::CaptionDisplayMode> fromString(const String& mode)
    {
        if (equalLettersIgnoringASCIICase(mode, "forcedonly"_s))
            return WebCore::CaptionUserPreferences::CaptionDisplayMode::ForcedOnly;
        if (equalLettersIgnoringASCIICase(mode, "manual"_s))
            return WebCore::CaptionUserPreferences::CaptionDisplayMode::Manual;
        if (equalLettersIgnoringASCIICase(mode, "automatic"_s))
            return WebCore::CaptionUserPreferences::CaptionDisplayMode::Automatic;
        if (equalLettersIgnoringASCIICase(mode, "alwayson"_s))
            return WebCore::CaptionUserPreferences::CaptionDisplayMode::AlwaysOn;
        return std::nullopt;
    }
};

} // namespace WTF

#endif
