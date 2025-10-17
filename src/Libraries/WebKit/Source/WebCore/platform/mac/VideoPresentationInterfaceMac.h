/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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

#if PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)

#include "HTMLMediaElementEnums.h"
#include "MediaPlayerIdentifier.h"
#include "PlaybackSessionInterfaceMac.h"
#include "PlaybackSessionModel.h"
#include "VideoFullscreenCaptions.h"
#include "VideoPresentationLayerProvider.h"
#include "VideoPresentationModel.h"
#include <wtf/CheckedRef.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSWindow;
OBJC_CLASS WebVideoPresentationInterfaceMacObjC;

namespace WebCore {

class IntRect;
class FloatSize;
class PlaybackSessionInterfaceMac;

class VideoPresentationInterfaceMac final
    : public VideoPresentationModelClient
    , private PlaybackSessionModelClient
    , public VideoFullscreenCaptions
    , public VideoPresentationLayerProvider
    , public RefCounted<VideoPresentationInterfaceMac>
    , public CanMakeCheckedPtr<VideoPresentationInterfaceMac> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(VideoPresentationInterfaceMac, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(VideoPresentationInterfaceMac);
public:
    static Ref<VideoPresentationInterfaceMac> create(PlaybackSessionInterfaceMac& playbackSessionInterface)
    {
        return adoptRef(*new VideoPresentationInterfaceMac(playbackSessionInterface));
    }
    ~VideoPresentationInterfaceMac();
    PlaybackSessionInterfaceMac& playbackSessionInterface() const { return m_playbackSessionInterface.get(); }
    RefPtr<VideoPresentationModel> videoPresentationModel() const { return m_videoPresentationModel.get(); }
    PlaybackSessionModel* playbackSessionModel() const { return m_playbackSessionInterface->playbackSessionModel(); }
    WEBCORE_EXPORT void setVideoPresentationModel(VideoPresentationModel*);

    // PlaybackSessionModelClient
    WEBCORE_EXPORT void rateChanged(OptionSet<PlaybackSessionModel::PlaybackState>, double playbackRate, double defaultPlaybackRate) override;
    WEBCORE_EXPORT void externalPlaybackChanged(bool  enabled, PlaybackSessionModel::ExternalPlaybackTargetType, const String& localizedDeviceName) override;
    WEBCORE_EXPORT void ensureControlsManager() override;

    // VideoPresentationModelClient
    WEBCORE_EXPORT void hasVideoChanged(bool) final;
    WEBCORE_EXPORT void videoDimensionsChanged(const FloatSize&) final;
    void setPlayerIdentifier(std::optional<MediaPlayerIdentifier> identifier) final { m_playerIdentifier = identifier; }

    WEBCORE_EXPORT void setupFullscreen(const IntRect& initialRect, NSWindow *parentWindow, HTMLMediaElementEnums::VideoFullscreenMode, bool allowsPictureInPicturePlayback);
    WEBCORE_EXPORT void enterFullscreen();
    WEBCORE_EXPORT bool exitFullscreen(const IntRect& finalRect, NSWindow *parentWindow);
    WEBCORE_EXPORT void exitFullscreenWithoutAnimationToMode(HTMLMediaElementEnums::VideoFullscreenMode);
    WEBCORE_EXPORT void cleanupFullscreen();
    WEBCORE_EXPORT void invalidate();
    void requestHideAndExitFullscreen() { }
    WEBCORE_EXPORT void preparedToReturnToInline(bool visible, const IntRect& inlineRect, NSWindow *parentWindow);
    void preparedToExitFullscreen() { }

    HTMLMediaElementEnums::VideoFullscreenMode mode() const { return m_mode; }
    bool hasMode(HTMLMediaElementEnums::VideoFullscreenMode mode) const { return m_mode & mode; }
    bool isMode(HTMLMediaElementEnums::VideoFullscreenMode mode) const { return m_mode == mode; }
    WEBCORE_EXPORT void setMode(HTMLMediaElementEnums::VideoFullscreenMode, bool);
    WEBCORE_EXPORT void clearMode(HTMLMediaElementEnums::VideoFullscreenMode);

    WEBCORE_EXPORT bool isPlayingVideoInEnhancedFullscreen() const;

    bool mayAutomaticallyShowVideoPictureInPicture() const { return false; }
    void applicationDidBecomeActive() { }

    WEBCORE_EXPORT WebVideoPresentationInterfaceMacObjC *videoPresentationInterfaceObjC();

    WEBCORE_EXPORT void requestHideAndExitPiP();

    std::optional<MediaPlayerIdentifier> playerIdentifier() const { return m_playerIdentifier; }

    WEBCORE_EXPORT void documentVisibilityChanged(bool) final;

    void swapFullscreenModesWith(VideoPresentationInterfaceMac&) { }

#if !RELEASE_LOG_DISABLED
    uint64_t logIdentifier() const;
    const Logger* loggerPtr() const;
    ASCIILiteral logClassName() const { return "VideoPresentationInterfaceMac"_s; };
    WTFLogChannel& logChannel() const;
#endif

private:
    WEBCORE_EXPORT VideoPresentationInterfaceMac(PlaybackSessionInterfaceMac&);

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }
    void setDocumentBecameVisibleCallback(Function<void()>&& callback) { m_documentBecameVisibleCallback = WTFMove(callback); }

    Ref<PlaybackSessionInterfaceMac> m_playbackSessionInterface;
    std::optional<MediaPlayerIdentifier> m_playerIdentifier;
    ThreadSafeWeakPtr<VideoPresentationModel> m_videoPresentationModel;
    HTMLMediaElementEnums::VideoFullscreenMode m_mode { HTMLMediaElementEnums::VideoFullscreenModeNone };
    RetainPtr<WebVideoPresentationInterfaceMacObjC> m_webVideoPresentationInterfaceObjC;
    bool m_documentIsVisible { true };
    Function<void()> m_documentBecameVisibleCallback;
};

} // namespace WebCore

#endif // PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)
