/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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
#import "WKDOMRangePrivate.h"

#import "InjectedBundleRangeHandle.h"
#import "WKBundleAPICast.h"
#import "WKDOMInternals.h"
#import <WebCore/Document.h>
#import <WebCore/TextIterator.h>
#import <WebCore/VisibleUnits.h>
#import <wtf/MainThread.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation WKDOMRange

- (id)_initWithImpl:(WebCore::Range*)impl
{
    self = [super init];
    if (!self)
        return nil;

    _impl = impl;
    WebKit::WKDOMRangeCache().add(impl, self);

    return self;
}

- (id)initWithDocument:(WKDOMDocument *)document
{
    return [self _initWithImpl:WebCore::Range::create(*WebKit::toWebCoreDocument(document)).ptr()];
}

- (void)dealloc
{
    ensureOnMainRunLoop([range = WTFMove(_impl)] {
        WebKit::WKDOMRangeCache().remove(range.get());
    });
    [super dealloc];
}

- (void)setStart:(WKDOMNode *)node offset:(int)offset
{
    if (!node)
        return;
    _impl->setStart(*WebKit::toWebCoreNode(node), offset);
}

- (void)setEnd:(WKDOMNode *)node offset:(int)offset
{
    if (!node)
        return;
    _impl->setEnd(*WebKit::toWebCoreNode(node), offset);
}

- (void)collapse:(BOOL)toStart
{
    _impl->collapse(toStart);
}

- (void)selectNode:(WKDOMNode *)node
{
    if (!node)
        return;
    _impl->selectNode(*WebKit::toWebCoreNode(node));
}

- (void)selectNodeContents:(WKDOMNode *)node
{
    if (!node)
        return;
    _impl->selectNodeContents(*WebKit::toWebCoreNode(node));
}

- (WKDOMNode *)startContainer
{
    return WebKit::toWKDOMNode(&_impl->startContainer());
}

- (NSInteger)startOffset
{
    return _impl->startOffset();
}

- (WKDOMNode *)endContainer
{
    return WebKit::toWKDOMNode(&_impl->endContainer());
}

- (NSInteger)endOffset
{
    return _impl->endOffset();
}

- (NSString *)text
{
    auto range = makeSimpleRange(*_impl);
    range.start.document().updateLayout();
    return plainText(range);
}

- (BOOL)isCollapsed
{
    return _impl->collapsed();
}

- (NSArray *)textRects
{
    auto range = makeSimpleRange(*_impl);
    range.start.document().updateLayout(WebCore::LayoutOptions::IgnorePendingStylesheets);
    return createNSArray(WebCore::RenderObject::absoluteTextRects(range)).autorelease();
}

- (WKDOMRange *)rangeByExpandingToWordBoundaryByCharacters:(NSUInteger)characters inDirection:(WKDOMRangeDirection)direction
{
    auto range = makeSimpleRange(*_impl);
    auto newRange = rangeExpandedByCharactersInDirectionAtWordBoundary(makeDeprecatedLegacyPosition(direction == WKDOMRangeDirectionForward ? range.end : range.start), characters, direction == WKDOMRangeDirectionForward ? WebCore::SelectionDirection::Forward : WebCore::SelectionDirection::Backward);
    return adoptNS([[WKDOMRange alloc] _initWithImpl:createLiveRange(newRange).get()]).autorelease();
}

@end

@implementation WKDOMRange (WKPrivate)

- (WKBundleRangeHandleRef)_copyBundleRangeHandleRef
{
    auto rangeHandle = WebKit::InjectedBundleRangeHandle::getOrCreate(_impl.get());
    return toAPI(rangeHandle.leakRef());
}

@end
