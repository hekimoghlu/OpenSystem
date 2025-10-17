/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#include "BitmapTexturePool.h"

#if USE(TEXTURE_MAPPER)

namespace WebCore {

#if defined(BITMAP_TEXTURE_POOL_MAX_SIZE_IN_MB) && BITMAP_TEXTURE_POOL_MAX_SIZE_IN_MB > 0
static constexpr size_t poolSizeLimit = BITMAP_TEXTURE_POOL_MAX_SIZE_IN_MB * MB;
#else
static constexpr size_t poolSizeLimit = std::numeric_limits<size_t>::max();
#endif

static const Seconds releaseUnusedSecondsTolerance { 3_s };
static const Seconds releaseUnusedTexturesTimerInterval { 500_ms };
static const Seconds releaseUnusedSecondsToleranceOnLimitExceeded { 50_ms };
static const Seconds releaseUnusedTexturesTimerIntervalOnLimitExceeded { 200_ms };

BitmapTexturePool::BitmapTexturePool()
    : m_releaseUnusedTexturesTimer(RunLoop::current(), this, &BitmapTexturePool::releaseUnusedTexturesTimerFired)
    , m_releaseUnusedSecondsTolerance(releaseUnusedSecondsTolerance)
    , m_releaseUnusedTexturesTimerInterval(releaseUnusedTexturesTimerInterval)
{
}

Ref<BitmapTexture> BitmapTexturePool::acquireTexture(const IntSize& size, OptionSet<BitmapTexture::Flags> flags)
{
    Entry* selectedEntry = std::find_if(m_textures.begin(), m_textures.end(),
        [&](Entry& entry) {
            return entry.m_texture->refCount() == 1
                && entry.m_texture->size() == size
                && entry.m_texture->flags().contains(BitmapTexture::Flags::DepthBuffer) == flags.contains(BitmapTexture::Flags::DepthBuffer);
        });

    if (selectedEntry == m_textures.end()) {
        m_textures.append(Entry(BitmapTexture::create(size, flags)));
        selectedEntry = &m_textures.last();
        m_poolSize += size.unclampedArea();
    } else
        selectedEntry->m_texture->reset(size, flags);

    enterLimitExceededModeIfNeeded();

    scheduleReleaseUnusedTextures();

    selectedEntry->markIsInUse();
    return selectedEntry->m_texture;
}

void BitmapTexturePool::scheduleReleaseUnusedTextures()
{
    if (m_releaseUnusedTexturesTimer.isActive())
        return;

    m_releaseUnusedTexturesTimer.startOneShot(m_releaseUnusedTexturesTimerInterval);
}

void BitmapTexturePool::releaseUnusedTexturesTimerFired()
{
    if (m_textures.isEmpty())
        return;

    // Delete entries, which have been unused in releaseUnusedSecondsTolerance.
    MonotonicTime minUsedTime = MonotonicTime::now() - m_releaseUnusedSecondsTolerance;

    m_textures.removeAllMatching([this, &minUsedTime](const Entry& entry) {
        if (entry.canBeReleased(minUsedTime)) {
            m_poolSize -= entry.m_texture->size().unclampedArea();
            return true;
        }
        return false;
    });

    exitLimitExceededModeIfNeeded();

    if (!m_textures.isEmpty())
        scheduleReleaseUnusedTextures();
}

void BitmapTexturePool::enterLimitExceededModeIfNeeded()
{
    if (m_onLimitExceededMode)
        return;

    if (m_poolSize > poolSizeLimit) {
        // If we allocated a new texture and this caused that we went over the size limit, enter limit exceeded mode,
        // set values for tolerance and interval for this mode, and trigger an immediate request to release textures.
        // While on limit exceeded mode, we are more aggressive releasing textures, by polling more often and keeping
        // the unused textures in the pool for smaller periods of time.
        m_onLimitExceededMode = true;
        m_releaseUnusedSecondsTolerance = releaseUnusedSecondsToleranceOnLimitExceeded;
        m_releaseUnusedTexturesTimerInterval = releaseUnusedTexturesTimerIntervalOnLimitExceeded;
        m_releaseUnusedTexturesTimer.startOneShot(0_s);
    }
}

void BitmapTexturePool::exitLimitExceededModeIfNeeded()
{
    if (!m_onLimitExceededMode)
        return;

    // If we're in limit exceeded mode and the pool size has become smaller than the limit,
    // exit the limit exceeded mode and set the default values for interval and tolerance again.
    if (m_poolSize <= poolSizeLimit) {
        m_onLimitExceededMode = false;
        m_releaseUnusedSecondsTolerance = releaseUnusedSecondsTolerance;
        m_releaseUnusedTexturesTimerInterval = releaseUnusedTexturesTimerInterval;
    }
}

} // namespace WebCore

#endif // USE(TEXTURE_MAPPER)
