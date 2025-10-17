/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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

#include "ImageTypes.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Image;
class IntRect;
class Settings;

// Interface for notification about changes to an image, including decoding,
// drawing, and animating.
class ImageObserver : public RefCountedAndCanMakeWeakPtr<ImageObserver> {
public:
    virtual ~ImageObserver() = default;

    virtual URL sourceUrl() const = 0;
    virtual String mimeType() const = 0;
    virtual unsigned numberOfClients() const { return 0; }
    virtual long long expectedContentLength() const = 0;

    virtual void encodedDataStatusChanged(const Image&, EncodedDataStatus) { };
    virtual void decodedSizeChanged(const Image&, long long delta) = 0;

    virtual void didDraw(const Image&) = 0;

    virtual bool canDestroyDecodedData(const Image&) const { return true; }
    virtual void imageFrameAvailable(const Image&, ImageAnimatingState, const IntRect* changeRect = nullptr, DecodingStatus = DecodingStatus::Invalid) = 0;
    virtual void changedInRect(const Image&, const IntRect* changeRect = nullptr) = 0;
    virtual void scheduleRenderingUpdate(const Image&) = 0;

    virtual bool allowsAnimation(const Image&) const { return true; }
    virtual const Settings* settings() { return nullptr; }

protected:
    ImageObserver() = default;
};

}
