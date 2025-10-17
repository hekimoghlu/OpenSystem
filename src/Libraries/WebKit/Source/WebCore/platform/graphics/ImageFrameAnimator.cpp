/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#include "config.h"
#include "ImageFrameAnimator.h"

#include "BitmapImageSource.h"
#include "Logging.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageFrameAnimator);

ImageFrameAnimator::ImageFrameAnimator(BitmapImageSource& source)
    : m_source(source)
    , m_frameCount(source.frameCount())
    , m_repetitionCount(source.repetitionCount())
    , m_currentFrameIndex(source.primaryFrameIndex())
{
    ASSERT(m_frameCount > 0);
    ASSERT(m_repetitionCount != RepetitionCountNone);
    ASSERT(m_currentFrameIndex < m_frameCount);
}

void ImageFrameAnimator::ref() const
{
    m_source.get()->ref();
}

void ImageFrameAnimator::deref() const
{
    m_source.get()->deref();
}

ImageFrameAnimator::~ImageFrameAnimator()
{
    clearTimer();
}

void ImageFrameAnimator::destroyDecodedData(bool destroyAll)
{
    // Animated images over a certain size are considered large enough that we'll
    // only hang on to one frame at a time.
    static constexpr unsigned LargeAnimationCutoff = 30 * 1024 * 1024;

    RefPtr source = m_source.get();
    if (source->decodedSize() < LargeAnimationCutoff)
        return;

    source->destroyDecodedData(destroyAll);
}

void ImageFrameAnimator::startTimer(Seconds delay)
{
    ASSERT(!m_frameTimer);
    m_frameTimer = makeUnique<Timer>(*this, &ImageFrameAnimator::timerFired);
    m_frameTimer->startOneShot(delay);
}

void ImageFrameAnimator::clearTimer()
{
    m_frameTimer = nullptr;
}

void ImageFrameAnimator::timerFired()
{
    clearTimer();

    RefPtr source = m_source.get();

    // Don't advance to nextFrame if the next frame is being decoded.
    if (source->isPendingDecodingAtIndex(nextFrameIndex(), m_nextFrameSubsamplingLevel, m_nextFrameOptions))
        return;

    advanceAnimation();
    source->imageFrameAtIndexAvailable(m_currentFrameIndex, ImageAnimatingState::Yes, source->frameDecodingStatusAtIndex(m_currentFrameIndex));
}

bool ImageFrameAnimator::imageFrameDecodeAtIndexHasFinished(unsigned index, ImageAnimatingState animatingState, DecodingStatus decodingStatus)
{
    if (animatingState != ImageAnimatingState::Yes)
        return false;

    if (index != nextFrameIndex())
        return false;

    if (!hasEverAnimated())
        return false;

    // Don't advance to nextFrame if the timer has not fired yet.
    if (isAnimating())
        return true;

    advanceAnimation();
    m_source.get()->imageFrameAtIndexAvailable(m_currentFrameIndex, animatingState, decodingStatus);
    return true;
}

bool ImageFrameAnimator::startAnimation(SubsamplingLevel subsamplingLevel, const DecodingOptions& options)
{
    if (m_frameTimer)
        return true;

    RefPtr source = m_source.get();

    // ImageObserver may disallow animation.
    if (!source->isAnimationAllowed())
        return false;

    m_nextFrameSubsamplingLevel = subsamplingLevel;
    m_nextFrameOptions = options;

    LOG(Images, "ImageFrameAnimator::%s - %p - url: %s. Animation at index = %d will be started.", __FUNCTION__, this, sourceUTF8().data(), m_currentFrameIndex);

    if (options.decodingMode() == DecodingMode::Asynchronous) {
        LOG(Images, "ImageFrameAnimator::%s - %p - url: %s. Decoding for frame at index = %d will be requested.", __FUNCTION__, this, sourceUTF8().data(), nextFrameIndex());
        source->requestNativeImageAtIndexIfNeeded(nextFrameIndex(), subsamplingLevel, ImageAnimatingState::Yes, options);
    }

    auto time = MonotonicTime::now();

    // Handle initial state.
    if (!m_desiredFrameStartTime)
        m_desiredFrameStartTime = time;

    auto duration = source->frameDurationAtIndex(m_currentFrameIndex);

    // Setting 'm_desiredFrameStartTime' to 'time' means we are late; otherwise we are early.
    m_desiredFrameStartTime = std::max(time, m_desiredFrameStartTime + duration);

    startTimer(m_desiredFrameStartTime - time);
    return true;
}

void ImageFrameAnimator::advanceAnimation()
{
    LOG(Images, "ImageFrameAnimator::%s - %p - url: %s. Animation at index = %d will be advanced.", __FUNCTION__, this, sourceUTF8().data(), m_currentFrameIndex);

    m_currentFrameIndex = nextFrameIndex();
    if (m_currentFrameIndex == m_frameCount - 1) {
        LOG(Images, "ImageFrameAnimator::%s - %p - url: %s. Animation loop %d has ended.", __FUNCTION__, this, sourceUTF8().data(), m_repetitionsComplete);
        ++m_repetitionsComplete;
    }

    destroyDecodedData(false);
}

void ImageFrameAnimator::stopAnimation()
{
    clearTimer();
}

void ImageFrameAnimator::resetAnimation()
{
    stopAnimation();

    m_currentFrameIndex = m_source.get()->primaryFrameIndex();
    m_repetitionsComplete = RepetitionCountNone;
    m_desiredFrameStartTime = { };

    // For extremely large animations, when the animation is reset, we just throw everything away.
    destroyDecodedData(true);
}

bool ImageFrameAnimator::isAnimationAllowed() const
{
    return m_repetitionCount == RepetitionCountInfinite || m_repetitionsComplete < m_repetitionCount;
}

CString ImageFrameAnimator::sourceUTF8() const
{
    return m_source.get()->sourceUTF8();
}

void ImageFrameAnimator::dump(TextStream& ts) const
{
    ts.dumpProperty("current-frame-index", m_currentFrameIndex);
    ts.dumpProperty("repetitions-complete", m_repetitionsComplete);
}

} // namespace WebCore
