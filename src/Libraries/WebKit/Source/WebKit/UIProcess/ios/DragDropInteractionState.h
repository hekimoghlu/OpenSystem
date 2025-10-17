/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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

#if PLATFORM(IOS_FAMILY) && ENABLE(DRAG_SUPPORT)

#import "UIKitSPI.h"
#import <WebCore/DragActions.h>
#import <WebCore/DragData.h>
#import <WebCore/Path.h>
#import <WebCore/TextIndicator.h>
#import <WebCore/WebItemProviderPasteboard.h>
#import <wtf/BlockPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/URL.h>
#import <wtf/Vector.h>

namespace WebCore {
struct DragItem;
struct TextIndicatorData;
}

namespace WebKit {

struct DragSourceState {
    OptionSet<WebCore::DragSourceAction> action;
    CGRect dragPreviewFrameInRootViewCoordinates { CGRectZero };
    RetainPtr<UIImage> image;
    std::optional<WebCore::TextIndicatorData> indicatorData;
    std::optional<WebCore::Path> visiblePath;
    String linkTitle;
    URL linkURL;
    bool possiblyNeedsDragPreviewUpdate { true };
    bool containsSelection { false };

    NSInteger itemIdentifier { 0 };
};

using DragItemToPreviewMap = HashMap<RetainPtr<UIDragItem>, RetainPtr<UITargetedDragPreview>>;

enum class AddPreviewViewToContainer : bool;

class DragDropInteractionState {
public:
    bool anyActiveDragSourceContainsSelection() const;

    // These helper methods are unique to UIDragInteraction.
    void prepareForDragSession(id <UIDragSession>, dispatch_block_t completionHandler);
    void dragSessionWillBegin();
    void stageDragItem(const WebCore::DragItem&, UIImage *);
    bool hasStagedDragSource() const;
    const DragSourceState& stagedDragSource() const { return m_stagedDragSource.value(); }
    enum class DidBecomeActive : bool { No, Yes };
    void clearStagedDragSource(DidBecomeActive = DidBecomeActive::No);
    UITargetedDragPreview *previewForLifting(UIDragItem *, UIView *contentView, UIView *previewContainer, const std::optional<WebCore::TextIndicatorData>&) const;
    UITargetedDragPreview *previewForCancelling(UIDragItem *, UIView *contentView, UIView *previewContainer);
    void dragSessionWillDelaySetDownAnimation(dispatch_block_t completion);
    bool shouldRequestAdditionalItemForDragSession(id <UIDragSession>) const;
    void dragSessionWillRequestAdditionalItem(void (^completion)(NSArray <UIDragItem *> *));

    void dropSessionDidEnterOrUpdate(id <UIDropSession>, const WebCore::DragData&);
    void dropSessionDidExit() { m_dropSession = nil; }
    void dropSessionWillPerformDrop() { m_isPerformingDrop = true; }

    void dragAndDropSessionsDidBecomeInactive();

    CGPoint adjustedPositionForDragEnd() const { return m_adjustedPositionForDragEnd; }
    bool didBeginDragging() const { return m_didBeginDragging; }
    bool isPerformingDrop() const { return m_isPerformingDrop; }
    id<UIDragSession> dragSession() const { return m_dragSession.get(); }
    id<UIDropSession> dropSession() const { return m_dropSession.get(); }
    BlockPtr<void()> takeDragStartCompletionBlock() { return WTFMove(m_dragStartCompletionBlock); }
    BlockPtr<void(NSArray<UIDragItem *> *)> takeAddDragItemCompletionBlock() { return WTFMove(m_addDragItemCompletionBlock); }
    Vector<RetainPtr<UIView>> takePreviewViewsForDragCancel() { return std::exchange(m_previewViewsForDragCancel, { }); }

    void addDefaultDropPreview(UIDragItem *, UITargetedDragPreview *);
    UITargetedDragPreview *finalDropPreview(UIDragItem *) const;
    void deliverDelayedDropPreview(UIView *contentView, UIView *previewContainer, const WebCore::TextIndicatorData&);
    void deliverDelayedDropPreview(UIView *contentView, CGRect unobscuredContentRect, NSArray<UIDragItem *> *, const Vector<WebCore::IntRect>& placeholderRects);

private:
    void updatePreviewsForActiveDragSources();
    std::optional<DragSourceState> activeDragSourceForItem(UIDragItem *) const;
    UITargetedDragPreview *defaultDropPreview(UIDragItem *) const;

    RetainPtr<UITargetedDragPreview> createDragPreviewInternal(UIDragItem *, UIView *contentView, UIView *previewContainer, AddPreviewViewToContainer, const std::optional<WebCore::TextIndicatorData>&) const;

    CGPoint m_lastGlobalPosition { CGPointZero };
    CGPoint m_adjustedPositionForDragEnd { CGPointZero };
    bool m_didBeginDragging { false };
    bool m_isPerformingDrop { false };
    RetainPtr<id <UIDragSession>> m_dragSession;
    RetainPtr<id <UIDropSession>> m_dropSession;
    BlockPtr<void()> m_dragStartCompletionBlock;
    BlockPtr<void(NSArray<UIDragItem *> *)> m_addDragItemCompletionBlock;
    Vector<RetainPtr<UIView>> m_previewViewsForDragCancel;

    std::optional<DragSourceState> m_stagedDragSource;
    Vector<DragSourceState> m_activeDragSources;
    DragItemToPreviewMap m_defaultDropPreviews;
    DragItemToPreviewMap m_finalDropPreviews;
};

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY) && ENABLE(DRAG_SUPPORT)
