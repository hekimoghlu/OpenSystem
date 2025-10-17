/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#import "WebPolicyDelegatePrivate.h"

#import <WebCore/FrameLoaderTypes.h>
#import <wtf/ObjCRuntimeExtras.h>

using namespace WebCore;

NSString *WebActionButtonKey = @"WebActionButtonKey"; 
NSString *WebActionElementKey = @"WebActionElementKey";
NSString *WebActionFormKey = @"WebActionFormKey";
NSString *WebActionModifierFlagsKey = @"WebActionModifierFlagsKey";
NSString *WebActionNavigationTypeKey = @"WebActionNavigationTypeKey";
NSString *WebActionOriginalURLKey = @"WebActionOriginalURLKey";

@interface WebPolicyDecisionListenerPrivate : NSObject
{
@public
    RetainPtr<id> target;
    SEL action;
}

- (id)initWithTarget:(id)target action:(SEL)action;

@end

@implementation WebPolicyDecisionListenerPrivate

- (id)initWithTarget:(id)t action:(SEL)a
{
    self = [super init];
    if (!self)
        return nil;
    target = t;
    action = a;
    return self;
}

@end

@implementation WebPolicyDecisionListener

- (id)_initWithTarget:(id)target action:(SEL)action
{
    self = [super init];
    if (!self)
        return nil;
    _private = [[WebPolicyDecisionListenerPrivate alloc] initWithTarget:target action:action];
    return self;
}

-(void)dealloc
{
    [_private release];
    [super dealloc];
}

- (void)_usePolicy:(PolicyAction)policy
{
    if (_private->target)
        wtfObjCMsgSend<void>(_private->target.get(), _private->action, policy);
}

- (void)_invalidate
{
    _private->target = nil;
}

// WebPolicyDecisionListener implementation

- (void)use
{
    [self _usePolicy:PolicyAction::Use];
}

- (void)ignore
{
    [self _usePolicy:PolicyAction::Ignore];
}

- (void)download
{
    [self _usePolicy:PolicyAction::Download];
}

@end
