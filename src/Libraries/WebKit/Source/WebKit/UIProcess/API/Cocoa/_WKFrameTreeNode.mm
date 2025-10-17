/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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

#import "WKFrameInfoInternal.h"
#import "WKSecurityOriginInternal.h"
#import "WKWebViewInternal.h"
#import "WebFrameProxy.h"
#import "WebPageProxy.h"
#import "_WKFrameHandleInternal.h"
#import "_WKFrameTreeNodeInternal.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKFrameTreeNode

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKFrameTreeNode.class, self))
        return;
    _node->API::FrameTreeNode::~FrameTreeNode();
    [super dealloc];
}

- (WKFrameInfo *)info
{
    return wrapper(API::FrameInfo::create(WebKit::FrameInfoData(_node->info()), &_node->page())).autorelease();
}

- (NSArray<_WKFrameTreeNode *> *)childFrames
{
    return createNSArray(_node->childFrames(), [&] (auto& child) {
        return wrapper(API::FrameTreeNode::create(WebKit::FrameTreeNodeData(child), _node->page()));
    }).autorelease();
}

- (API::Object&)_apiObject
{
    return *_node;
}

@end
