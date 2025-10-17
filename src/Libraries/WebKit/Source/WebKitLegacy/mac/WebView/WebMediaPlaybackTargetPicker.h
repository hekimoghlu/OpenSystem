/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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

#if ENABLE(WIRELESS_PLAYBACK_TARGET) && !PLATFORM(IOS_FAMILY)

#include <WebCore/MediaPlaybackTarget.h>
#include <WebCore/MediaPlaybackTargetContext.h>
#include <WebCore/WebMediaSessionManagerClient.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakObjCPtr.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS WebView;

namespace WebCore {
class FloatRect;
class MediaPlaybackTarget;
class Page;
}

class WebMediaPlaybackTargetPicker : public WebCore::WebMediaSessionManagerClient {
    WTF_MAKE_TZONE_ALLOCATED(WebMediaPlaybackTargetPicker);
public:
    static std::unique_ptr<WebMediaPlaybackTargetPicker> create(WebView *, WebCore::Page&);

    explicit WebMediaPlaybackTargetPicker(WebView *, WebCore::Page&);
    virtual ~WebMediaPlaybackTargetPicker() = default;

    void addPlaybackTargetPickerClient(WebCore::PlaybackTargetClientContextIdentifier);
    void removePlaybackTargetPickerClient(WebCore::PlaybackTargetClientContextIdentifier);
    void showPlaybackTargetPicker(WebCore::PlaybackTargetClientContextIdentifier, const WebCore::FloatRect&, bool hasVideo);
    void playbackTargetPickerClientStateDidChange(WebCore::PlaybackTargetClientContextIdentifier, WebCore::MediaProducerMediaStateFlags);
    void setMockMediaPlaybackTargetPickerEnabled(bool);
    void setMockMediaPlaybackTargetPickerState(const String&, WebCore::MediaPlaybackTargetContext::MockState);
    void mockMediaPlaybackTargetPickerDismissPopup();

    void invalidate();

private:
    // WebMediaSessionManagerClient
    void setPlaybackTarget(WebCore::PlaybackTargetClientContextIdentifier, Ref<WebCore::MediaPlaybackTarget>&&) final;
    void externalOutputDeviceAvailableDidChange(WebCore::PlaybackTargetClientContextIdentifier, bool) final;
    void setShouldPlayToPlaybackTarget(WebCore::PlaybackTargetClientContextIdentifier, bool) final;
    void playbackTargetPickerWasDismissed(WebCore::PlaybackTargetClientContextIdentifier) final;
    RetainPtr<PlatformView> platformView() const final;

    WeakPtr<WebCore::Page> m_page;
    WeakObjCPtr<WebView> m_webView;
};

#endif
