/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
#import "_WKFrameHandleInternal.h"

#import <WebCore/FrameIdentifier.h>
#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKFrameHandle

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKFrameHandle.class, self))
        return;

    _frameHandle->~FrameHandle();

    [super dealloc];
}

- (BOOL)isEqual:(id)object
{
    if (object == self)
        return YES;

    auto *handle = dynamic_objc_cast<_WKFrameHandle>(object);
    if (!handle)
        return NO;

    return _frameHandle->frameID() == handle->_frameHandle->frameID();
}

- (NSUInteger)hash
{
    return _frameHandle->frameID() ? _frameHandle->frameID()->object().toUInt64() : 0;
}

- (uint64_t)frameID
{
    return _frameHandle->frameID() ? _frameHandle->frameID()->object().toUInt64() : 0;
}

#pragma mark NSCopying protocol implementation

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark NSSecureCoding protocol implementation

+ (BOOL)supportsSecureCoding
{
    return YES;
}

- (id)initWithCoder:(NSCoder *)decoder
{
    if (!(self = [super init]))
        return nil;

    NSNumber *frameID = [decoder decodeObjectOfClass:[NSNumber class] forKey:@"frameID"];
    if (![frameID isKindOfClass:[NSNumber class]]) {
        [self release];
        return nil;
    }

    auto rawFrameID = frameID.unsignedLongLongValue;
    if (!ObjectIdentifier<WebCore::FrameIdentifierType>::isValidIdentifier(rawFrameID)) {
        [self release];
        return nil;
    }

    NSNumber *processID = [decoder decodeObjectOfClass:[NSNumber class] forKey:@"processID"];
    if (![processID isKindOfClass:[NSNumber class]]) {
        [self release];
        return nil;
    }

    auto rawProcessID = processID.unsignedLongLongValue;
    if (!ObjectIdentifier<WebCore::ProcessIdentifierType>::isValidIdentifier(rawProcessID)) {
        [self release];
        return nil;
    }

    API::Object::constructInWrapper<API::FrameHandle>(self, WebCore::FrameIdentifier {
        ObjectIdentifier<WebCore::FrameIdentifierType>(rawFrameID),
        ObjectIdentifier<WebCore::ProcessIdentifierType>(rawProcessID)
    }, false);

    return self;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
    [coder encodeObject:@(_frameHandle->frameID() ? _frameHandle->frameID()->object().toUInt64() : 0) forKey:@"frameID"];
    [coder encodeObject:@(_frameHandle->frameID() ? _frameHandle->frameID()->processIdentifier().toUInt64() : 0) forKey:@"processID"];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_frameHandle;
}

@end
