/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#import "_WKTargetedElementInfo.h"

#import "APIFrameTreeNode.h"
#import "WKObject.h"
#import "WebPageProxy.h"
#import "_WKFrameTreeNodeInternal.h"
#import "_WKTargetedElementInfoInternal.h"
#import <WebCore/ShareableBitmap.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/BlockPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKTargetedElementInfo

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKTargetedElementInfo.class, self))
        return;
    _info->API::TargetedElementInfo::~TargetedElementInfo();
    [super dealloc];
}

- (API::Object&)_apiObject
{
    return *_info;
}

- (_WKTargetedElementPosition)positionType
{
    switch (_info->positionType()) {
    case WebCore::PositionType::Static:
        return _WKTargetedElementPositionStatic;
    case WebCore::PositionType::Relative:
        return _WKTargetedElementPositionRelative;
    case WebCore::PositionType::Absolute:
        return _WKTargetedElementPositionAbsolute;
    case WebCore::PositionType::Sticky:
        return _WKTargetedElementPositionSticky;
    case WebCore::PositionType::Fixed:
        return _WKTargetedElementPositionFixed;
    }
}

- (CGRect)bounds
{
    return _info->boundsInRootView();
}

- (CGRect)boundsInWebView
{
    return _info->boundsInWebView();
}

- (CGRect)boundsInClientCoordinates
{
    return _info->boundsInClientCoordinates();
}

- (NSArray<NSString *> *)selectors
{
    if (_info->selectors().isEmpty())
        return @[ ];

    if (_info->isInShadowTree())
        return @[ ];

    return createNSArray(_info->selectors().first()).autorelease();
}

- (NSArray<NSArray<NSString *> *> *)selectorsIncludingShadowHosts
{
    RetainPtr result = adoptNS([[NSMutableArray alloc] initWithCapacity:_info->selectors().size()]);
    for (auto& selectors : _info->selectors()) {
        RetainPtr nsSelectors = adoptNS([[NSMutableArray alloc] initWithCapacity:selectors.size()]);
        for (auto& selector : selectors)
            [nsSelectors addObject:selector];
        [result addObject:nsSelectors.get()];
    }
    return result.autorelease();
}

- (NSString *)renderedText
{
    return _info->renderedText();
}

- (NSString *)searchableText
{
    return _info->searchableText();
}

- (NSString *)screenReaderText
{
    return _info->screenReaderText();
}

- (_WKRectEdge)offsetEdges
{
    _WKRectEdge edges = _WKRectEdgeNone;
    auto coreEdges = _info->offsetEdges();
    if (coreEdges.top())
        edges |= _WKRectEdgeTop;
    if (coreEdges.left())
        edges |= _WKRectEdgeLeft;
    if (coreEdges.bottom())
        edges |= _WKRectEdgeBottom;
    if (coreEdges.right())
        edges |= _WKRectEdgeRight;
    return edges;
}

- (void)getChildFrames:(void(^)(NSArray<_WKFrameTreeNode *> *))completion
{
    return _info->childFrames([completion = makeBlockPtr(completion)](auto&& nodes) {
        completion(createNSArray(WTFMove(nodes), [](API::FrameTreeNode& node) {
            return wrapper(node);
        }).autorelease());
    });
}

- (BOOL)isSameElement:(_WKTargetedElementInfo *)other
{
    return _info->isSameElement(*other->_info);
}

- (BOOL)isNearbyTarget
{
    return _info->isNearbyTarget();
}

- (BOOL)isInVisibilityAdjustmentSubtree
{
    return _info->isInVisibilityAdjustmentSubtree();
}

- (BOOL)hasLargeReplacedDescendant
{
    return _info->hasLargeReplacedDescendant();
}

- (NSSet<NSURL *> *)mediaAndLinkURLs
{
    RetainPtr result = adoptNS([NSMutableSet<NSURL *> new]);
    for (auto& url : _info->mediaAndLinkURLs())
        [result addObject:(NSURL *)url];
    return result.autorelease();
}

- (BOOL)isPseudoElement
{
    return _info->isPseudoElement();
}

- (BOOL)isInShadowTree
{
    return _info->isInShadowTree();
}

- (BOOL)hasAudibleMedia
{
    return _info->hasAudibleMedia();
}

- (void)takeSnapshotWithCompletionHandler:(void(^)(CGImageRef))completion
{
    return _info->takeSnapshot([completion = makeBlockPtr(completion)](std::optional<WebCore::ShareableBitmapHandle>&& imageHandle) mutable {
        if (!imageHandle)
            return completion(nullptr);

        if (RefPtr bitmap = WebCore::ShareableBitmap::create(WTFMove(*imageHandle), WebCore::SharedMemory::Protection::ReadOnly))
            return completion(bitmap->makeCGImage().get());

        completion(nullptr);
    });
}

- (NSString *)debugDescription
{
    auto firstSelector = [&]() -> String {
        auto& allSelectors = _info->selectors();
        if (allSelectors.isEmpty())
            return { };

        if (allSelectors.last().isEmpty())
            return { };

        // Most relevant selector for the final target element (after all enclosing shadow
        // roots have been resolved).
        return allSelectors.last().first();
    }();

    auto bounds = _info->boundsInRootView();
    return [NSString stringWithFormat:@"<%@ %p \"%@\" at {{%.0f,%.0f},{%.0f,%.0f}}>"
        , self.class, self, (NSString *)firstSelector
        , bounds.x(), bounds.y(), bounds.width(), bounds.height()];
}

@end
