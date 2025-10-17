/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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

#if ENABLE(VIDEO_PRESENTATION_MODE)

#include "EventListener.h"
#include "FloatRect.h"
#include "HTMLMediaElement.h"
#include "MediaPlayerEnums.h"
#include "MediaPlayerIdentifier.h"
#include "PlatformLayer.h"
#include "VideoPresentationModel.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Function.h>
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class AudioTrack;
class HTMLVideoElement;
class TextTrack;
class PlaybackSessionModelMediaElement;

enum class AudioSessionCategory : uint8_t;
enum class AudioSessionMode : uint8_t;
enum class RouteSharingPolicy : uint8_t;

class VideoPresentationModelVideoElement final
    : public VideoPresentationModel
    , public HTMLMediaElementClient {
public:
    void ref() const final { VideoPresentationModel::ref(); }
    void deref() const final { VideoPresentationModel::deref(); }

    static Ref<VideoPresentationModelVideoElement> create()
    {
        return adoptRef(*new VideoPresentationModelVideoElement());
    }
    WEBCORE_EXPORT ~VideoPresentationModelVideoElement();
    WEBCORE_EXPORT void setVideoElement(HTMLVideoElement*);
    HTMLVideoElement* videoElement() const { return m_videoElement.get(); }
    WEBCORE_EXPORT RetainPtr<PlatformLayer> createVideoFullscreenLayer();
    WEBCORE_EXPORT void setVideoFullscreenLayer(PlatformLayer*, Function<void()>&& completionHandler = [] { });
    WEBCORE_EXPORT void willExitFullscreen() final;
    WEBCORE_EXPORT void waitForPreparedForInlineThen(Function<void()>&& completionHandler);

    WEBCORE_EXPORT void addClient(VideoPresentationModelClient&) final;
    WEBCORE_EXPORT void removeClient(VideoPresentationModelClient&) final;
    WEBCORE_EXPORT void requestFullscreenMode(HTMLMediaElementEnums::VideoFullscreenMode, bool finishedWithMedia = false) final;
    WEBCORE_EXPORT void setVideoLayerFrame(FloatRect) final;
    WEBCORE_EXPORT void setVideoLayerGravity(MediaPlayerEnums::VideoGravity) final;
    WEBCORE_EXPORT void setVideoFullscreenFrame(FloatRect) final;
    WEBCORE_EXPORT void fullscreenModeChanged(HTMLMediaElementEnums::VideoFullscreenMode) final;
    FloatSize videoDimensions() const final { return m_videoDimensions; }
    bool hasVideo() const final { return m_hasVideo; }

    WEBCORE_EXPORT void setVideoSizeFenced(const FloatSize&, WTF::MachSendRight&&);

    WEBCORE_EXPORT void requestRouteSharingPolicyAndContextUID(CompletionHandler<void(RouteSharingPolicy, String)>&&) final;
    WEBCORE_EXPORT void setRequiresTextTrackRepresentation(bool) final;
    WEBCORE_EXPORT void setTextTrackRepresentationBounds(const IntRect&) final;

#if !RELEASE_LOG_DISABLED
    const Logger* loggerPtr() const final;
    WEBCORE_EXPORT uint64_t logIdentifier() const final;
    WEBCORE_EXPORT uint64_t nextChildIdentifier() const final;
    ASCIILiteral logClassName() const { return "VideoPresentationModelVideoElement"_s; }
    WTFLogChannel& logChannel() const;
#endif

protected:
    WEBCORE_EXPORT VideoPresentationModelVideoElement();

private:
    class VideoListener final : public EventListener {
    public:
        static Ref<VideoListener> create(VideoPresentationModelVideoElement& parent)
        {
            return adoptRef(*new VideoListener(parent));
        }
        void handleEvent(WebCore::ScriptExecutionContext&, WebCore::Event&) final;
    private:
        explicit VideoListener(VideoPresentationModelVideoElement&);

        ThreadSafeWeakPtr<VideoPresentationModelVideoElement> m_parent;
    };

    void setHasVideo(bool);
    void setVideoDimensions(const FloatSize&);
    void setPlayerIdentifier(std::optional<MediaPlayerIdentifier>);

    void willEnterPictureInPicture() final;
    void didEnterPictureInPicture() final;
    void failedToEnterPictureInPicture() final;
    void willExitPictureInPicture() final;
    void didExitPictureInPicture() final;

    static std::span<const AtomString> observedEventNames();
    static std::span<const AtomString> documentObservedEventNames();
    const AtomString& eventNameAll();
    friend class VideoListener;
    void updateForEventName(const AtomString&);
    void cleanVideoListeners();
    void documentVisibilityChanged();

    // HTMLMediaElementClient
    void audioSessionCategoryChanged(AudioSessionCategory, AudioSessionMode, RouteSharingPolicy) final;

    Ref<VideoListener> m_videoListener;
    RefPtr<HTMLVideoElement> m_videoElement;
    RetainPtr<PlatformLayer> m_videoFullscreenLayer;
    bool m_isListening { false };
    UncheckedKeyHashSet<CheckedPtr<VideoPresentationModelClient>> m_clients;
    bool m_hasVideo { false };
    bool m_documentIsVisible { true };
    FloatSize m_videoDimensions;
    FloatRect m_videoFrame;
    Vector<RefPtr<TextTrack>> m_legibleTracksForMenu;
    Vector<RefPtr<AudioTrack>> m_audioTracksForMenu;
    std::optional<MediaPlayerIdentifier> m_playerIdentifier;

#if !RELEASE_LOG_DISABLED
    mutable uint64_t m_childIdentifierSeed { 0 };
#endif
};

} // namespace WebCore

#endif // ENABLE(VIDEO_PRESENTATION_MODE)
