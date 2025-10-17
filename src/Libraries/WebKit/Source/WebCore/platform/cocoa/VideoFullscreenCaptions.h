/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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

#include "PlatformImage.h"

OBJC_CLASS CALayer;

namespace WebCore {

class FloatSize;

class VideoFullscreenCaptions {
public:
    WEBCORE_EXPORT virtual ~VideoFullscreenCaptions();

    WEBCORE_EXPORT void setTrackRepresentationImage(PlatformImagePtr textTrack);
    WEBCORE_EXPORT void setTrackRepresentationContentsScale(float);
    WEBCORE_EXPORT void setTrackRepresentationHidden(bool);

    WEBCORE_EXPORT virtual CALayer* captionsLayer();
    WEBCORE_EXPORT void setCaptionsFrame(const CGRect&);
    WEBCORE_EXPORT virtual void setupCaptionsLayer(CALayer *parent, const FloatSize&);
    WEBCORE_EXPORT void removeCaptionsLayer();

protected:
    WEBCORE_EXPORT VideoFullscreenCaptions();

    bool m_captionsLayerHidden  { false };
    RetainPtr<CALayer> m_captionsLayer;
    RetainPtr<id> m_captionsLayerContents;
};

}
