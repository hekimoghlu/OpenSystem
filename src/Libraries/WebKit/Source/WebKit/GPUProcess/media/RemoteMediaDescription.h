/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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

#if ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)

#include "MediaDescriptionInfo.h"

namespace WebKit {

class RemoteMediaDescription : public WebCore::MediaDescription {
public:
    static Ref<RemoteMediaDescription> create(const MediaDescriptionInfo& descriptionInfo)
    {
        return adoptRef(*new RemoteMediaDescription(descriptionInfo));
    }

    virtual ~RemoteMediaDescription() = default;

    bool isVideo() const final { return m_isVideo; }
    bool isAudio() const final { return m_isAudio; }
    bool isText() const final { return m_isText;}

private:
    RemoteMediaDescription(const MediaDescriptionInfo& descriptionInfo)
        : MediaDescription(descriptionInfo.m_codec.isolatedCopy())
        , m_isVideo(descriptionInfo.m_isVideo)
        , m_isAudio(descriptionInfo.m_isAudio)
        , m_isText(descriptionInfo.m_isText)
    {
    }

    bool m_isVideo { false };
    bool m_isAudio { false };
    bool m_isText { false };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)
