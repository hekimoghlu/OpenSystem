/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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
#import "config.h"
#import "MemoryRelease.h"

#import "CGSubimageCacheWithTimer.h"
#import "FontCache.h"
#import "GCController.h"
#import "HTMLNameCache.h"
#import "IOSurfacePool.h"
#import "LayerPool.h"
#import "LocaleCocoa.h"
#import <notify.h>
#import <pal/spi/ios/GraphicsServicesSPI.h>

#if PLATFORM(IOS_FAMILY)
#import "LegacyTileCache.h"
#import "TileControllerMemoryHandlerIOS.h"
#endif


namespace WebCore {

void platformReleaseMemory(Critical)
{
#if PLATFORM(IOS_FAMILY) && !PLATFORM(IOS_FAMILY_SIMULATOR) && !PLATFORM(MACCATALYST)
    // FIXME: Remove this call to GSFontInitialize() once <rdar://problem/32886715> is fixed.
    GSFontInitialize();
    GSFontPurgeFontCache();
#endif

    LocaleCocoa::releaseMemory();

    for (auto& pool : LayerPool::allLayerPools())
        pool->drain();

#if PLATFORM(IOS_FAMILY)
    LegacyTileCache::drainLayerPool();
    tileControllerMemoryHandler().trimUnparentedTilesToTarget(0);
#endif

    IOSurfacePool::sharedPoolSingleton().discardAllSurfaces();

#if CACHE_SUBIMAGES
    CGSubimageCacheWithTimer::clear();
#endif
}

void platformReleaseGraphicsMemory(Critical)
{
    IOSurfacePool::sharedPoolSingleton().discardAllSurfaces();

#if CACHE_SUBIMAGES
    CGSubimageCacheWithTimer::clear();
#endif
}

void jettisonExpensiveObjectsOnTopLevelNavigation()
{
    // Protect against doing excessive jettisoning during repeated navigations.
    const auto minimumTimeSinceNavigation = 2_s;

    static auto timeOfLastNavigation = MonotonicTime::now();
    auto now = MonotonicTime::now();
    bool shouldJettison = now - timeOfLastNavigation >= minimumTimeSinceNavigation;
    timeOfLastNavigation = now;

    if (!shouldJettison)
        return;

#if PLATFORM(IOS_FAMILY)
    // Throw away linked JS code. Linked code is tied to a global object and is not reusable.
    // The immediate memory savings outweigh the cost of recompilation in case we go back again.
    GCController::singleton().deleteAllLinkedCode(JSC::DeleteAllCodeIfNotCollecting);
#endif

    HTMLNameCache::clear();
}

void registerMemoryReleaseNotifyCallbacks()
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        int dummy;
        notify_register_dispatch("com.apple.WebKit.fullGC", &dummy, dispatch_get_main_queue(), ^(int) {
            GCController::singleton().garbageCollectNow();
        });
        notify_register_dispatch("com.apple.WebKit.deleteAllCode", &dummy, dispatch_get_main_queue(), ^(int) {
            GCController::singleton().deleteAllCode(JSC::PreventCollectionAndDeleteAllCode);
            GCController::singleton().garbageCollectNow();
        });
    });
}

} // namespace WebCore
