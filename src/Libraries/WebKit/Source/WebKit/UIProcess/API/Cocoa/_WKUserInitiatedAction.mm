/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
#import "_WKUserInitiatedActionInternal.h"

#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKUserInitiatedAction

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;
    API::Object::constructInWrapper<API::UserInitiatedAction>(self);
    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKUserInitiatedAction.class, self))
        return;

    _userInitiatedAction->~UserInitiatedAction();

    [super dealloc];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; consumed = %s>", NSStringFromClass(self.class), self, self.consumed ? "YES" : "NO"];
}

- (void)consume
{
    _userInitiatedAction->setConsumed();
}

- (BOOL)isConsumed
{
    return _userInitiatedAction->consumed();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_userInitiatedAction;
}

@end
