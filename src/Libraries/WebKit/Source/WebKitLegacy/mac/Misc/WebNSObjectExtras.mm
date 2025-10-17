/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
#import "WebNSObjectExtras.h"

#import <wtf/Assertions.h>
#import <wtf/RetainPtr.h>
#import <wtf/text/StringCommon.h>

@interface WebMainThreadInvoker : NSProxy
{
    id target;
    RetainPtr<id> exception;
}
@end

@interface NSInvocation (WebMainThreadInvoker)
- (void)_webkit_invokeAndHandleException:(WebMainThreadInvoker *)exceptionHandler;
@end

static bool returnTypeIsObject(NSInvocation *invocation)
{
    // Could use either _C_ID or NSObjCObjectType, but it seems that neither is
    // both available and non-deprecated on all versions of Mac OS X we support.
    return contains(unsafeSpan([[invocation methodSignature] methodReturnType]), '@');
}

@implementation WebMainThreadInvoker

- (id)initWithTarget:(id)passedTarget
{
    target = passedTarget;
    return self;
}

- (void)forwardInvocation:(NSInvocation *)invocation
{
    [invocation setTarget:target];
    [invocation performSelectorOnMainThread:@selector(_webkit_invokeAndHandleException:) withObject:self waitUntilDone:YES];
    if (exception) {
        auto exceptionToThrow = std::exchange(exception, nil);
        @throw exceptionToThrow.autorelease();
    } else if (returnTypeIsObject(invocation)) {
        // _webkit_invokeAndHandleException retained the return value on the main thread.
        // Now autorelease it on the calling thread.
        id returnValue;
        [invocation getReturnValue:&returnValue];
        adoptNS(returnValue).autorelease();
    }
}

- (NSMethodSignature *)methodSignatureForSelector:(SEL)selector
{
    return [target methodSignatureForSelector:selector];
}

- (void)handleException:(id)passedException
{
    ASSERT(!exception);
    exception = passedException;
}

@end

@implementation NSInvocation (WebMainThreadInvoker)

- (void)_webkit_invokeAndHandleException:(WebMainThreadInvoker *)exceptionHandler
{
    @try {
        [self invoke];
    } @catch (id exception) {
        [exceptionHandler handleException:exception];
        return;
    }
    if (returnTypeIsObject(self)) {
        // Retain the return value on the main thread.
        // -[WebMainThreadInvoker forwardInvocation:] will autorelease it on the calling thread.
        id value;
        [self getReturnValue:&value];
        [value retain];
    }
}

@end

@implementation NSObject (WebNSObjectExtras)

+ (id)_webkit_invokeOnMainThread
{
    return adoptNS([[WebMainThreadInvoker alloc] initWithTarget:self]).autorelease();
}

- (id)_webkit_invokeOnMainThread
{
    return adoptNS([[WebMainThreadInvoker alloc] initWithTarget:self]).autorelease();
}

@end
