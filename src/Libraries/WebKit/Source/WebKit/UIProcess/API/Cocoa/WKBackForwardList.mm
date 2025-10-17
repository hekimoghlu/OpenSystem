/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#import "WKBackForwardListInternal.h"

#import "WKBackForwardListItemInternal.h"
#import "WKNSArray.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKBackForwardList {
    API::ObjectStorage<WebKit::WebBackForwardList> _list;
}

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKBackForwardList.class, self))
        return;

    _list->~WebBackForwardList();

    [super dealloc];
}

- (WKBackForwardListItem *)currentItem
{
    return WebKit::wrapper(_list->currentItem());
}

- (WKBackForwardListItem *)backItem
{
    return WebKit::wrapper(_list->backItem());
}

- (WKBackForwardListItem *)forwardItem
{
    return WebKit::wrapper(_list->forwardItem());
}

- (WKBackForwardListItem *)itemAtIndex:(NSInteger)index
{
    return WebKit::wrapper(_list->itemAtIndex(index));
}

- (NSArray *)backList
{
    return WebKit::wrapper(_list->backList()).autorelease();
}

- (NSArray *)forwardList
{
    return WebKit::wrapper(_list->forwardList()).autorelease();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_list;
}

@end

@implementation WKBackForwardList (WKPrivate)

- (void)_removeAllItems
{
    _list->removeAllItems();
}

- (void)_clear
{
    _list->clear();
}

@end
