/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#import "WKRemoteObject.h"

#import "_WKRemoteObjectInterface.h"
#import "_WKRemoteObjectRegistryInternal.h"
#import <objc/runtime.h>
#import <wtf/RetainPtr.h>

@implementation WKRemoteObject {
    RetainPtr<_WKRemoteObjectRegistry> _objectRegistry;
    RetainPtr<_WKRemoteObjectInterface> _interface;
}

- (instancetype)_initWithObjectRegistry:(_WKRemoteObjectRegistry *)objectRegistry interface:(_WKRemoteObjectInterface *)interface
{
    if (!(self = [super init]))
        return nil;

    _objectRegistry = objectRegistry;
    _interface = interface;

    return self;
}

- (BOOL)conformsToProtocol:(Protocol *)protocol
{
    return [super conformsToProtocol:protocol] || protocol_conformsToProtocol([_interface protocol], protocol);
}

static const char* methodArgumentTypeEncodingForSelector(Protocol *protocol, SEL selector)
{
    // First look at required methods.
    struct objc_method_description method = protocol_getMethodDescription(protocol, selector, YES, YES);
    if (method.name)
        return method.types;

    // Then look at optional methods.
    method = protocol_getMethodDescription(protocol, selector, NO, YES);
    if (method.name)
        return method.types;

    return nullptr;
}

- (NSMethodSignature *)methodSignatureForSelector:(SEL)selector
{
    if (!selector)
        return nil;

    Protocol *protocol = [_interface protocol];

    const char* types = methodArgumentTypeEncodingForSelector(protocol, selector);
    if (!types) {
        // We didn't find anything the protocol, fall back to the superclass.
        return [super methodSignatureForSelector:selector];
    }

    return [NSMethodSignature signatureWithObjCTypes:types];
}

- (void)forwardInvocation:(NSInvocation *)invocation
{
    [_objectRegistry _sendInvocation:invocation interface:_interface.get()];
}

@end
