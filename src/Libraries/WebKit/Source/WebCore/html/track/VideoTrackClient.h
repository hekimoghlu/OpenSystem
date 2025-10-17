/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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

#include <wtf/WeakPtr.h>

namespace WebCore {
class VideoTrackClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::VideoTrackClient> : std::true_type { };
}

namespace WebCore {

class VideoTrack;

class VideoTrackClient : public CanMakeWeakPtr<VideoTrackClient> {
public:
    virtual ~VideoTrackClient() = default;
    virtual void videoTrackIdChanged(VideoTrack&) { }
    virtual void videoTrackKindChanged(VideoTrack&) { }
    virtual void videoTrackLabelChanged(VideoTrack&) { }
    virtual void videoTrackLanguageChanged(VideoTrack&) { }
    virtual void videoTrackSelectedChanged(VideoTrack&) { }
    virtual void willRemoveVideoTrack(VideoTrack&) { }
};

} // namespace WebCore

#endif

