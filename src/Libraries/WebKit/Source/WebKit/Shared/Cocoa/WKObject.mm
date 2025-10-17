/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#import "WKObject.h"

#import "APIObject.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/RetainPtr.h>

@interface NSObject ()
- (BOOL)isNSArray__;
- (BOOL)isNSCFConstantString__;
- (BOOL)isNSData__;
- (BOOL)isNSDate__;
- (BOOL)isNSDictionary__;
- (BOOL)isNSObject__;
- (BOOL)isNSOrderedSet__;
- (BOOL)isNSNumber__;
- (BOOL)isNSSet__;
- (BOOL)isNSString__;
- (BOOL)isNSTimeZone__;
- (BOOL)isNSValue__;
@end

@implementation WKObject {
    BOOL _hasInitializedTarget;
    RetainPtr<NSObject> _target;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKObject.class, self))
        return;

    API::Object::fromWKObjectExtraSpace(self).~Object();

    [super dealloc];
}

static inline void initializeTargetIfNeeded(WKObject *self)
{
    if (self->_hasInitializedTarget)
        return;

    self->_hasInitializedTarget = YES;
    self->_target = adoptNS([self _web_createTarget]);
}

- (BOOL)isEqual:(id)object
{
    if (object == self)
        return YES;

    if (!object)
        return NO;

    initializeTargetIfNeeded(self);

    return [_target isEqual:object];
}

- (NSUInteger)hash
{
    initializeTargetIfNeeded(self);

    return _target ? [_target hash] : [super hash];
}

- (BOOL)isKindOfClass:(Class)aClass
{
    initializeTargetIfNeeded(self);

    return [_target isKindOfClass:aClass];
}

- (BOOL)isMemberOfClass:(Class)aClass
{
    initializeTargetIfNeeded(self);

    return [_target isMemberOfClass:aClass];
}

- (BOOL)respondsToSelector:(SEL)selector
{
    initializeTargetIfNeeded(self);

    return [_target respondsToSelector:selector] || [super respondsToSelector:selector];
}

- (BOOL)conformsToProtocol:(Protocol *)protocol
{
    initializeTargetIfNeeded(self);

    return [_target conformsToProtocol:protocol] || [super conformsToProtocol:protocol];
}

- (id)forwardingTargetForSelector:(SEL)selector
{
    initializeTargetIfNeeded(self);

    return _target.get();
}

- (NSString *)description
{
    initializeTargetIfNeeded(self);

    return _target ? [_target description] : [super description];
}

- (NSString *)debugDescription
{
    initializeTargetIfNeeded(self);

    return _target ? [_target debugDescription] : [self description];
}

- (Class)classForCoder
{
    initializeTargetIfNeeded(self);

    return [_target classForCoder];
}

- (Class)classForKeyedArchiver
{
    initializeTargetIfNeeded(self);

    return [_target classForKeyedArchiver];
}

- (NSObject *)_web_createTarget
{
    return nil;
}

- (void)forwardInvocation:(NSInvocation *)invocation
{
    initializeTargetIfNeeded(self);

    [invocation invokeWithTarget:_target.get()];
}

- (NSMethodSignature *)methodSignatureForSelector:(SEL)sel
{
    initializeTargetIfNeeded(self);

    return [_target methodSignatureForSelector:sel];
}

- (BOOL)isNSObject__
{
    initializeTargetIfNeeded(self);

    return [_target isNSObject__];
}

- (BOOL)isNSArray__
{
    initializeTargetIfNeeded(self);

    return [_target isNSArray__];
}

- (BOOL)isNSCFConstantString__
{
    initializeTargetIfNeeded(self);

    return [_target isNSCFConstantString__];
}

- (BOOL)isNSData__
{
    initializeTargetIfNeeded(self);

    return [_target isNSData__];
}

- (BOOL)isNSDate__
{
    initializeTargetIfNeeded(self);

    return [_target isNSDate__];
}

- (BOOL)isNSDictionary__
{
    initializeTargetIfNeeded(self);

    return [_target isNSDictionary__];
}

- (BOOL)isNSNumber__
{
    initializeTargetIfNeeded(self);

    return [_target isNSNumber__];
}

- (BOOL)isNSOrderedSet__
{
    initializeTargetIfNeeded(self);

    return [_target isNSOrderedSet__];
}

- (BOOL)isNSSet__
{
    initializeTargetIfNeeded(self);

    return [_target isNSSet__];
}

- (BOOL)isNSString__
{
    initializeTargetIfNeeded(self);

    return [_target isNSString__];
}

- (BOOL)isNSTimeZone__
{
    initializeTargetIfNeeded(self);

    return [_target isNSTimeZone__];
}

- (BOOL)isNSValue__
{
    initializeTargetIfNeeded(self);

    return [_target isNSValue__];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return API::Object::fromWKObjectExtraSpace(self);
}

@end
