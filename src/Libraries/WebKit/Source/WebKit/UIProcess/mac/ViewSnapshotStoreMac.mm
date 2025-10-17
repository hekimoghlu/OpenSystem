/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
#import "ViewSnapshotStore.h"

#import <CoreGraphics/CoreGraphics.h>
#import <WebCore/IOSurface.h>
#import <WebCore/ImageBuffer.h>

#if PLATFORM(IOS_FAMILY)
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#endif

namespace WebKit {

Ref<ViewSnapshot> ViewSnapshot::create(std::unique_ptr<WebCore::IOSurface> surface)
{
    return adoptRef(*new ViewSnapshot(WTFMove(surface)));
}

ViewSnapshot::ViewSnapshot(std::unique_ptr<WebCore::IOSurface> surface)
    : m_surface(WTFMove(surface))
{
    if (hasImage())
        ViewSnapshotStore::singleton().didAddImageToSnapshot(*this);
}

void ViewSnapshot::setSurface(std::unique_ptr<WebCore::IOSurface> surface)
{
    ASSERT(!m_surface);
    if (!surface) {
        clearImage();
        return;
    }

    m_surface = WTFMove(surface);
    ViewSnapshotStore::singleton().didAddImageToSnapshot(*this);
}

bool ViewSnapshot::hasImage() const
{
    return !!m_surface;
}

void ViewSnapshot::clearImage()
{
    if (!hasImage())
        return;

    ViewSnapshotStore::singleton().willRemoveImageFromSnapshot(*this);

    m_surface = nullptr;
}

WebCore::SetNonVolatileResult ViewSnapshot::setVolatile(bool becomeVolatile)
{
    if (ViewSnapshotStore::singleton().disableSnapshotVolatilityForTesting())
        return WebCore::SetNonVolatileResult::Valid;

    if (!m_surface)
        return WebCore::SetNonVolatileResult::Empty;
    return m_surface->setVolatile(becomeVolatile);
}

id ViewSnapshot::asLayerContents()
{
    if (!m_surface)
        return nullptr;

    if (setVolatile(false) != WebCore::SetNonVolatileResult::Valid) {
        clearImage();
        return nullptr;
    }

    return m_surface->asLayerContents();
}

RetainPtr<CGImageRef> ViewSnapshot::asImageForTesting()
{
    if (!m_surface)
        return nullptr;

    ASSERT(ViewSnapshotStore::singleton().disableSnapshotVolatilityForTesting());
    // Note: here we will destroy the context immediately, which will read back
    // the image to CPU. This should be fine for testing.
    auto context = m_surface->createPlatformContext();
    return m_surface->createImage(context.get());
}

} // namespace WebKit
