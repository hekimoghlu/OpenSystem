/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#import "WKDOMNodePrivate.h"

#import "InjectedBundleNodeHandle.h"
#import "WKBundleAPICast.h"
#import "WKDOMInternals.h"
#import <WebCore/Document.h>
#import <WebCore/RenderObject.h>
#import <WebCore/SimpleRange.h>
#import <wtf/MainThread.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation WKDOMNode

- (id)_initWithImpl:(WebCore::Node*)impl
{
    self = [super init];
    if (!self)
        return nil;

    _impl = impl;
    WebKit::WKDOMNodeCache().add(impl, self);

    return self;
}

- (void)dealloc
{
    ensureOnMainRunLoop([node = WTFMove(_impl)] {
        WebKit::WKDOMNodeCache().remove(node.get());
    });
    [super dealloc];
}

- (void)insertNode:(WKDOMNode *)node before:(WKDOMNode *)refNode
{
    if (!node)
        return;

    _impl->insertBefore(*WebKit::toWebCoreNode(node), WebKit::toWebCoreNode(refNode));
}

- (void)appendChild:(WKDOMNode *)node
{
    if (!node)
        return;

    _impl->appendChild(*WebKit::toWebCoreNode(node));
}

- (void)removeChild:(WKDOMNode *)node
{
    if (!node)
        return;

    _impl->removeChild(*WebKit::toWebCoreNode(node));
}

- (WKDOMDocument *)document
{
    return WebKit::toWKDOMDocument(&_impl->document());
}

- (WKDOMNode *)parentNode
{
    return WebKit::toWKDOMNode(_impl->parentNode());
}

- (WKDOMNode *)firstChild
{
    return WebKit::toWKDOMNode(_impl->firstChild());
}

- (WKDOMNode *)lastChild
{
    return WebKit::toWKDOMNode(_impl->lastChild());
}

- (WKDOMNode *)previousSibling
{
    return WebKit::toWKDOMNode(_impl->previousSibling());
}

- (WKDOMNode *)nextSibling
{
    return WebKit::toWKDOMNode(_impl->nextSibling());
}

- (NSArray *)textRects
{
    _impl->document().updateLayout(WebCore::LayoutOptions::IgnorePendingStylesheets);
    if (!_impl->renderer())
        return nil;
    return createNSArray(WebCore::RenderObject::absoluteTextRects(WebCore::makeRangeSelectingNodeContents(*_impl))).autorelease();
}

@end

@implementation WKDOMNode (WKPrivate)

- (WKBundleNodeHandleRef)_copyBundleNodeHandleRef
{
    return toAPI(WebKit::InjectedBundleNodeHandle::getOrCreate(_impl.get()).leakRef());
}

@end
