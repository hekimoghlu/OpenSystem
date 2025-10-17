/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#import "_WKContentRuleListActionInternal.h"

#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKContentRuleListAction

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKContentRuleListAction.class, self))
        return;

    _action->~ContentRuleListAction();
    
    [super dealloc];
}

- (BOOL)blockedLoad
{
#if ENABLE(CONTENT_EXTENSIONS)
    return _action->blockedLoad();
#else
    return NO;
#endif
}

- (BOOL)blockedCookies
{
#if ENABLE(CONTENT_EXTENSIONS)
    return _action->blockedCookies();
#else
    return NO;
#endif
}

- (BOOL)madeHTTPS
{
#if ENABLE(CONTENT_EXTENSIONS)
    return _action->madeHTTPS();
#else
    return NO;
#endif
}

- (BOOL)redirected
{
#if ENABLE(CONTENT_EXTENSIONS)
    return _action->redirected();
#else
    return NO;
#endif
}

- (BOOL)modifiedHeaders
{
#if ENABLE(CONTENT_EXTENSIONS)
    return _action->modifiedHeaders();
#else
    return NO;
#endif
}

- (NSArray<NSString *> *)notifications
{
#if ENABLE(CONTENT_EXTENSIONS)
    auto& vector = _action->notifications();
    if (vector.isEmpty())
        return nil;
    return createNSArray(vector).autorelease();
#else
    return nil;
#endif
}

- (API::Object&)_apiObject
{
    return *_action;
}

@end
