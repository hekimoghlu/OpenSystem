/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#include "DecodingOptions.h"
#include "ImageTypes.h"
#include "Timer.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class BitmapImageSource;
class ImageFrame;

class ImageFrameAnimator {
    WTF_MAKE_TZONE_ALLOCATED(ImageFrameAnimator);
public:
    explicit ImageFrameAnimator(BitmapImageSource&);

    void ref() const;
    void deref() const;

    ~ImageFrameAnimator();

    bool imageFrameDecodeAtIndexHasFinished(unsigned index, ImageAnimatingState, DecodingStatus);

    bool startAnimation(SubsamplingLevel, const DecodingOptions&);
    void advanceAnimation();
    void stopAnimation();
    void resetAnimation();
    bool isAnimating() const { return !!m_frameTimer; }
    bool isAnimationAllowed() const;

    bool hasEverAnimated() const { return !!m_desiredFrameStartTime; }
    unsigned currentFrameIndex() const { return m_currentFrameIndex; }

    void dump(TextStream&) const;

private:
    void destroyDecodedData(bool destroyAll);

    void startTimer(Seconds delay);
    void clearTimer();
    void timerFired();

    unsigned nextFrameIndex() const { return (m_currentFrameIndex + 1) % m_frameCount; }

    CString sourceUTF8() const;

    ThreadSafeWeakPtr<BitmapImageSource> m_source; // Cannot be null.
    unsigned m_frameCount { 0 };
    RepetitionCount m_repetitionCount { RepetitionCountNone };

    std::unique_ptr<Timer> m_frameTimer;
    SubsamplingLevel m_nextFrameSubsamplingLevel { SubsamplingLevel::Default };
    DecodingOptions m_nextFrameOptions { DecodingMode::Asynchronous };

    unsigned m_currentFrameIndex { 0 };
    RepetitionCount m_repetitionsComplete { RepetitionCountNone };
    MonotonicTime m_desiredFrameStartTime;
};

} // namespace WebCore
