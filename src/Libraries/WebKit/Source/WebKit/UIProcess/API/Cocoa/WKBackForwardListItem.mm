/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#import "WKBackForwardListItemInternal.h"

#import "WKNSURLExtras.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKBackForwardListItem {
    API::ObjectStorage<WebKit::WebBackForwardListItem> _item;
}

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKBackForwardListItem.class, self))
        return;

    _item->~WebBackForwardListItem();

    [super dealloc];
}

- (NSURL *)URL
{
    return [NSURL _web_URLWithWTFString:_item->url()];
}

- (NSString *)title
{
    if (!_item->title())
        return nil;

    return _item->title();
}

- (NSURL *)initialURL
{
    return [NSURL _web_URLWithWTFString:_item->originalURL()];
}

- (WebKit::WebBackForwardListItem&)_item
{
    return *_item;
}

- (CGImageRef)_copySnapshotForTesting
{
    if (auto snapshot = _item->snapshot())
        return snapshot->asImageForTesting().leakRef();
    return nullptr;
}

- (CGPoint)_scrollPosition
{
    return CGPointMake(_item->mainFrameState()->scrollPosition.x(), _item->mainFrameState()->scrollPosition.y());
}

- (BOOL)_wasCreatedByJSWithoutUserInteraction
{
    return _item->wasCreatedByJSWithoutUserInteraction();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_item;
}

@end
