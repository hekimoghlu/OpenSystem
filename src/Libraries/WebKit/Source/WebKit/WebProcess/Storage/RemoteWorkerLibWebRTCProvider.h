/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

#include "RTCDataChannelRemoteManager.h"

#if USE(LIBWEBRTC)
#include <WebCore/LibWebRTCProvider.h>
#elif USE(GSTREAMER_WEBRTC)
#include <WebCore/GStreamerWebRTCProvider.h>
#endif

namespace WebKit {

#if ENABLE(WEB_RTC)

#if USE(LIBWEBRTC)
using LibWebRTCProviderBase = WebCore::LibWebRTCProvider;
#else
using LibWebRTCProviderBase = WebCore::GStreamerWebRTCProvider;
#endif

class RemoteWorkerLibWebRTCProvider final : public LibWebRTCProviderBase {
public:
    RemoteWorkerLibWebRTCProvider() = default;

private:
    RefPtr<WebCore::RTCDataChannelRemoteHandlerConnection> createRTCDataChannelRemoteHandlerConnection() final { return &RTCDataChannelRemoteManager::singleton().remoteHandlerConnection(); }
};
#endif

} // namespace WebKit
