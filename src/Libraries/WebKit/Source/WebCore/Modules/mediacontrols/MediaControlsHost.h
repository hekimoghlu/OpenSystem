/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#include "HTMLMediaElement.h"
#include "MediaSession.h"
#include <variant>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AudioTrack;
class AudioTrackList;
class Element;
class WeakPtrImplWithEventTargetData;
class HTMLElement;
class HTMLMediaElement;
class MediaControlTextTrackContainerElement;
class TextTrack;
class TextTrackList;
class TextTrackRepresentation;
class VoidCallback;

class MediaControlsHost final
    : public RefCounted<MediaControlsHost>
#if ENABLE(MEDIA_SESSION)
    , private MediaSessionObserver
#endif
    , public CanMakeWeakPtr<MediaControlsHost> {
    WTF_MAKE_FAST_ALLOCATED(MediaControlsHost);
public:
    USING_CAN_MAKE_WEAKPTR(CanMakeWeakPtr<MediaControlsHost>);

    static Ref<MediaControlsHost> create(HTMLMediaElement&);
    ~MediaControlsHost();

    static const AtomString& automaticKeyword();
    static const AtomString& forcedOnlyKeyword();

    String layoutTraitsClassName() const;
    const AtomString& mediaControlsContainerClassName() const;

    double brightness() const { return 1; }
    void setBrightness(double) { }

    Vector<RefPtr<TextTrack>> sortedTrackListForMenu(TextTrackList&);
    Vector<RefPtr<AudioTrack>> sortedTrackListForMenu(AudioTrackList&);

    using TextOrAudioTrack = std::variant<RefPtr<TextTrack>, RefPtr<AudioTrack>>;
    String displayNameForTrack(const std::optional<TextOrAudioTrack>&);

    static TextTrack& captionMenuOffItem();
    static TextTrack& captionMenuAutomaticItem();
    AtomString captionDisplayMode() const;
    void setSelectedTextTrack(TextTrack*);
    Element* textTrackContainer();
    void updateTextTrackContainer();
    TextTrackRepresentation* textTrackRepresentation() const;
    bool allowsInlineMediaPlayback() const;
    bool supportsFullscreen() const;
    bool isVideoLayerInline() const;
    bool isInMediaDocument() const;
    bool userGestureRequired() const;
    bool shouldForceControlsDisplay() const;
    bool supportsSeeking() const;
    bool inWindowFullscreen() const;
    bool supportsRewind() const;
    bool needsChromeMediaControlsPseudoElement() const;

    enum class ForceUpdate : bool { No, Yes };
    void updateCaptionDisplaySizes(ForceUpdate = ForceUpdate::No);
    void updateTextTrackRepresentationImageIfNeeded();
    void enteredFullscreen();
    void exitedFullscreen();
    void requiresTextTrackRepresentationChanged();

    String externalDeviceDisplayName() const;

    enum class DeviceType { None, Airplay, Tvout };
    DeviceType externalDeviceType() const;

    bool controlsDependOnPageScaleFactor() const;
    void setControlsDependOnPageScaleFactor(bool v);

    static String generateUUID();

    static String shadowRootCSSText();
    static String base64StringForIconNameAndType(const String& iconName, const String& iconType);
    static String formattedStringForDuration(double);
#if ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS)
    bool showMediaControlsContextMenu(HTMLElement&, String&& optionsJSONString, Ref<VoidCallback>&&);
#endif // ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS)

    using SourceType = HTMLMediaElement::SourceType;
    std::optional<SourceType> sourceType() const;

    void presentationModeChanged();

#if ENABLE(MEDIA_SESSION)
    void ensureMediaSessionObserver();
#endif

private:
    explicit MediaControlsHost(HTMLMediaElement&);

    void savePreviouslySelectedTextTrackIfNecessary();
    void restorePreviouslySelectedTextTrackIfNecessary();

#if ENABLE(MEDIA_SESSION)
    RefPtr<MediaSession> mediaSession() const;

    // MediaSessionObserver
    void metadataChanged(const RefPtr<MediaMetadata>&) final;
#endif

    WeakPtr<HTMLMediaElement> m_mediaElement;
    RefPtr<MediaControlTextTrackContainerElement> m_textTrackContainer;
    RefPtr<TextTrack> m_previouslySelectedTextTrack;

#if ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS)
    RefPtr<VoidCallback> m_showMediaControlsContextMenuCallback;
#endif // ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS)
};

} // namespace WebCore

#endif // ENABLE(VIDEO)

