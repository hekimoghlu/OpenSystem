/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
#include "InjectedBundleRangeHandle.h"

#include "InjectedBundleNodeHandle.h"
#include "WebImage.h"
#include <JavaScriptCore/APICast.h>
#include <JavaScriptCore/HeapInlines.h>
#include <WebCore/Document.h>
#include <WebCore/FloatRect.h>
#include <WebCore/FrameSelection.h>
#include <WebCore/GeometryUtilities.h>
#include <WebCore/GraphicsContext.h>
#include <WebCore/IntRect.h>
#include <WebCore/JSRange.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/LocalFrameView.h>
#include <WebCore/Page.h>
#include <WebCore/Range.h>
#include <WebCore/RenderView.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/SimpleRange.h>
#include <WebCore/TextIterator.h>
#include <WebCore/VisibleSelection.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

#if PLATFORM(MAC)
#include <WebCore/LocalDefaultSystemAppearance.h>
#endif

namespace WebKit {
using namespace WebCore;

using DOMRangeHandleCache = HashMap<SingleThreadWeakRef<WebCore::Range>, WeakRef<InjectedBundleRangeHandle>>;

static DOMRangeHandleCache& domRangeHandleCache()
{
    static NeverDestroyed<DOMRangeHandleCache> cache;
    return cache;
}

RefPtr<InjectedBundleRangeHandle> InjectedBundleRangeHandle::getOrCreate(JSContextRef context, JSObjectRef object)
{
    return getOrCreate(JSRange::toWrapped(toJS(context)->vm(), toJS(object)));
}

RefPtr<InjectedBundleRangeHandle> InjectedBundleRangeHandle::getOrCreate(WebCore::Range* range)
{
    if (!range)
        return nullptr;

    RefPtr<InjectedBundleRangeHandle> newRange;
    auto result = domRangeHandleCache().ensure(*range, [&] {
        newRange = adoptRef(*new InjectedBundleRangeHandle(*range));
        return WeakRef { *newRange };
    });
    return newRange ? newRange.releaseNonNull() : Ref { result.iterator->value.get() };
}

InjectedBundleRangeHandle::InjectedBundleRangeHandle(WebCore::Range& range)
    : m_range(range)
{
}

InjectedBundleRangeHandle::~InjectedBundleRangeHandle()
{
    domRangeHandleCache().remove(m_range.get());
}

WebCore::Range& InjectedBundleRangeHandle::coreRange() const
{
    return m_range.get();
}

Ref<InjectedBundleNodeHandle> InjectedBundleRangeHandle::document()
{
    return InjectedBundleNodeHandle::getOrCreate(m_range->startContainer().document());
}

WebCore::IntRect InjectedBundleRangeHandle::boundingRectInWindowCoordinates() const
{
    auto range = makeSimpleRange(m_range);
    auto frame = range.start.document().frame();
    if (!frame)
        return { };
    auto view = frame->view();
    if (!view)
        return { };
    return view->contentsToWindow(enclosingIntRect(unionRectIgnoringZeroRects(RenderObject::absoluteBorderAndTextRects(range))));
}

RefPtr<WebImage> InjectedBundleRangeHandle::renderedImage(SnapshotOptions options)
{
    auto range = makeSimpleRange(m_range);

    Ref document = range.start.document();

    RefPtr frame = document->frame();
    if (!frame)
        return nullptr;

    auto frameView = frame->view();
    if (!frameView)
        return nullptr;

#if PLATFORM(MAC)
    LocalDefaultSystemAppearance localAppearance(frameView->useDarkAppearance());
#endif

    VisibleSelection oldSelection = frame->selection().selection();
    frame->selection().setSelection(range);

    float scaleFactor = options.contains(SnapshotOption::ExcludeDeviceScaleFactor) ? 1 : frame->page()->deviceScaleFactor();
    IntRect paintRect = enclosingIntRect(unionRectIgnoringZeroRects(RenderObject::absoluteBorderAndTextRects(range)));
    IntSize backingStoreSize = paintRect.size();
    backingStoreSize.scale(scaleFactor);

    auto snapshot = WebImage::create(backingStoreSize, snapshotOptionsToImageOptions(options | SnapshotOption::Shareable), DestinationColorSpace::SRGB());
    if (!snapshot->context())
        return nullptr;

    auto& graphicsContext = *snapshot->context();
    graphicsContext.scale(scaleFactor);

    paintRect.move(frameView->frameRect().x(), frameView->frameRect().y());
    paintRect.moveBy(-frameView->scrollPosition());

    graphicsContext.translate(-paintRect.location());

    OptionSet<PaintBehavior> oldPaintBehavior = frameView->paintBehavior();
    OptionSet<PaintBehavior> paintBehavior = oldPaintBehavior;
    paintBehavior.add({ PaintBehavior::SelectionOnly, PaintBehavior::FlattenCompositingLayers, PaintBehavior::Snapshotting });
    if (options.contains(SnapshotOption::ForceBlackText))
        paintBehavior.add(PaintBehavior::ForceBlackText);
    if (options.contains(SnapshotOption::ForceWhiteText))
        paintBehavior.add(PaintBehavior::ForceWhiteText);

    frameView->setPaintBehavior(paintBehavior);
    document->updateLayout();

    frameView->paint(graphicsContext, paintRect);
    frameView->setPaintBehavior(oldPaintBehavior);

    frame->selection().setSelection(oldSelection);

    return snapshot;
}

String InjectedBundleRangeHandle::text() const
{
    auto range = makeSimpleRange(m_range);
    range.start.protectedDocument()->updateLayout();
    return plainText(range);
}

RefPtr<InjectedBundleRangeHandle> createHandle(const std::optional<WebCore::SimpleRange>& range)
{
    return InjectedBundleRangeHandle::getOrCreate(createLiveRange(range).get());
}

} // namespace WebKit
