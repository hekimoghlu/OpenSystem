/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#import "_WKWebPushMessageInternal.h"

#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/SpanCocoa.h>

@implementation _WKWebPushMessage

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKWebPushMessage.class, self))
        return;

    _message->API::WebPushMessage::~WebPushMessage();
    [super dealloc];
}

- (NSData *)data
{
    if (auto messageData = self._protectedMessage->data())
        return toNSData(*messageData).get();

    return nil;
}

- (NSURL *)scope
{
    return _message->scope();
}

- (NSString *)partition
{
    return _message->partition();
}

- (API::Object&)_apiObject
{
    return *_message;
}

- (Ref<API::WebPushMessage>)_protectedMessage
{
    return *_message;
}

@end

