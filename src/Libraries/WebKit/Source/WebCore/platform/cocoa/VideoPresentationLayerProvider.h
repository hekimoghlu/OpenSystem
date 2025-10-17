/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

#include "PlatformView.h"

OBJC_CLASS WebAVPlayerLayer;
OBJC_CLASS WebAVPlayerLayerView;

namespace WebCore {

class VideoPresentationLayerProvider {
public:
    WEBCORE_EXPORT virtual ~VideoPresentationLayerProvider();

    PlatformView *layerHostView() const { return m_layerHostView.get(); }
    void setLayerHostView(RetainPtr<PlatformView>&& layerHostView) { m_layerHostView = WTFMove(layerHostView); }

    WebAVPlayerLayer *playerLayer() const { return m_playerLayer.get(); }
    virtual void setPlayerLayer(RetainPtr<WebAVPlayerLayer>&& layer) { m_playerLayer = WTFMove(layer); }

#if PLATFORM(IOS_FAMILY)
    WebAVPlayerLayerView *playerLayerView() const { return m_playerLayerView.get(); }
    void setPlayerLayerView(RetainPtr<WebAVPlayerLayerView>&& playerLayerView) { m_playerLayerView = WTFMove(playerLayerView); }

    PlatformView *videoView() const { return m_videoView.get(); }
    void setVideoView(RetainPtr<PlatformView>&& videoView) { m_videoView = WTFMove(videoView); }
#endif

protected:
    WEBCORE_EXPORT VideoPresentationLayerProvider();

private:
    RetainPtr<PlatformView> m_layerHostView;
    RetainPtr<WebAVPlayerLayer> m_playerLayer;

#if PLATFORM(IOS_FAMILY)
    RetainPtr<WebAVPlayerLayerView> m_playerLayerView;
    RetainPtr<PlatformView> m_videoView;
#endif
};

}

#endif
