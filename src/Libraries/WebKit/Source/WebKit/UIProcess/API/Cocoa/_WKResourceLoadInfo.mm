/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
#import "_WKResourceLoadInfo.h"

#import "APIFrameHandle.h"
#import "APIResourceLoadInfo.h"
#import "ResourceLoadInfo.h"
#import "_WKFrameHandleInternal.h"
#import "_WKResourceLoadInfoInternal.h"
#import <WebCore/WebCoreObjCExtras.h>

static _WKResourceLoadInfoResourceType toWKResourceLoadInfoResourceType(WebKit::ResourceLoadInfo::Type type)
{
    using namespace WebKit;
    switch (type) {
    case ResourceLoadInfo::Type::ApplicationManifest:
        return _WKResourceLoadInfoResourceTypeApplicationManifest;
    case ResourceLoadInfo::Type::Beacon:
        return _WKResourceLoadInfoResourceTypeBeacon;
    case ResourceLoadInfo::Type::CSPReport:
        return _WKResourceLoadInfoResourceTypeCSPReport;
    case ResourceLoadInfo::Type::Document:
        return _WKResourceLoadInfoResourceTypeDocument;
    case ResourceLoadInfo::Type::Fetch:
        return _WKResourceLoadInfoResourceTypeFetch;
    case ResourceLoadInfo::Type::Font:
        return _WKResourceLoadInfoResourceTypeFont;
    case ResourceLoadInfo::Type::Image:
        return _WKResourceLoadInfoResourceTypeImage;
    case ResourceLoadInfo::Type::Media:
        return _WKResourceLoadInfoResourceTypeMedia;
    case ResourceLoadInfo::Type::Object:
        return _WKResourceLoadInfoResourceTypeObject;
    case ResourceLoadInfo::Type::Other:
        return _WKResourceLoadInfoResourceTypeOther;
    case ResourceLoadInfo::Type::Ping:
        return _WKResourceLoadInfoResourceTypePing;
    case ResourceLoadInfo::Type::Script:
        return _WKResourceLoadInfoResourceTypeScript;
    case ResourceLoadInfo::Type::Stylesheet:
        return _WKResourceLoadInfoResourceTypeStylesheet;
    case ResourceLoadInfo::Type::XMLHTTPRequest:
        return _WKResourceLoadInfoResourceTypeXMLHTTPRequest;
    case ResourceLoadInfo::Type::XSLT:
        return _WKResourceLoadInfoResourceTypeXSLT;
    }
    
    ASSERT_NOT_REACHED();
    return _WKResourceLoadInfoResourceTypeOther;
}


@implementation _WKResourceLoadInfo

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKResourceLoadInfo.class, self))
        return;
    _info->API::ResourceLoadInfo::~ResourceLoadInfo();
    [super dealloc];
}

- (uint64_t)resourceLoadID
{
    return _info->resourceLoadID().toUInt64();
}

- (_WKFrameHandle *)frame
{
    if (auto frameID = _info->frameID())
        return wrapper(API::FrameHandle::create(*frameID)).autorelease();
    return nil;
}

- (_WKFrameHandle *)parentFrame
{
    if (auto parentFrameID = _info->parentFrameID())
        return wrapper(API::FrameHandle::create(*parentFrameID)).autorelease();
    return nil;
}

- (NSUUID *)documentID
{
    if (auto documentID = _info->documentID())
        return documentID.value();
    return nil;
}

- (NSURL *)originalURL
{
    return _info->originalURL();
}

- (NSString *)originalHTTPMethod
{
    return _info->originalHTTPMethod();
}

- (NSDate *)eventTimestamp
{
    return [NSDate dateWithTimeIntervalSince1970:_info->eventTimestamp().secondsSinceEpoch().seconds()];
}

- (BOOL)loadedFromCache
{
    return _info->loadedFromCache();
}

- (_WKResourceLoadInfoResourceType)resourceType
{
    return toWKResourceLoadInfoResourceType(_info->resourceLoadType());
}

- (API::Object&)_apiObject
{
    return *_info;
}

+ (BOOL)supportsSecureCoding
{
    return YES;
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    if (!(self = [super init]))
        return nil;

    NSNumber *resourceLoadID = [coder decodeObjectOfClass:[NSNumber class] forKey:@"resourceLoadID"];
    if (!resourceLoadID) {
        [self release];
        return nil;
    }

    _WKFrameHandle *frame = [coder decodeObjectOfClass:[_WKFrameHandle class] forKey:@"frame"];
    if (!frame) {
        [self release];
        return nil;
    }

    _WKFrameHandle *parentFrame = [coder decodeObjectOfClass:[_WKFrameHandle class] forKey:@"parentFrame"];
    // parentFrame is nullable, so decoding null is ok.

    NSUUID *documentID = [coder decodeObjectOfClass:NSUUID.class forKey:@"documentID"];
    // documentID is nullable, so decoding null is ok.

    NSURL *originalURL = [coder decodeObjectOfClass:[NSURL class] forKey:@"originalURL"];
    if (!originalURL) {
        [self release];
        return nil;
    }

    NSString *originalHTTPMethod = [coder decodeObjectOfClass:[NSString class] forKey:@"originalHTTPMethod"];
    if (!originalHTTPMethod) {
        [self release];
        return nil;
    }

    NSDate *eventTimestamp = [coder decodeObjectOfClass:[NSDate class] forKey:@"eventTimestamp"];
    if (!eventTimestamp) {
        [self release];
        return nil;
    }

    NSNumber *loadedFromCache = [coder decodeObjectOfClass:[NSNumber class] forKey:@"loadedFromCache"];
    if (!loadedFromCache) {
        [self release];
        return nil;
    }

    NSNumber *type = [coder decodeObjectOfClass:[NSNumber class] forKey:@"type"];
    if (!type) {
        [self release];
        return nil;
    }

    WebKit::ResourceLoadInfo info {
        ObjectIdentifier<WebKit::NetworkResourceLoadIdentifierType>(resourceLoadID.unsignedLongLongValue),
        frame->_frameHandle->frameID(),
        parentFrame ? parentFrame->_frameHandle->frameID() : std::nullopt,
        documentID ? WTF::UUID::fromNSUUID(documentID) : std::nullopt,
        originalURL,
        originalHTTPMethod,
        WallTime::fromRawSeconds(eventTimestamp.timeIntervalSince1970),
        static_cast<bool>(loadedFromCache.boolValue),
        static_cast<WebKit::ResourceLoadInfo::Type>(type.unsignedCharValue),
    };

    API::Object::constructInWrapper<API::ResourceLoadInfo>(self, WTFMove(info));

    return self;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
    [coder encodeObject:@(self.resourceLoadID) forKey:@"resourceLoadID"];
    [coder encodeObject:self.frame forKey:@"frame"];
    [coder encodeObject:self.parentFrame forKey:@"parentFrame"];
    [coder encodeObject:self.documentID forKey:@"documentID"];
    [coder encodeObject:self.originalURL forKey:@"originalURL"];
    [coder encodeObject:self.originalHTTPMethod forKey:@"originalHTTPMethod"];
    [coder encodeObject:self.eventTimestamp forKey:@"eventTimestamp"];
    [coder encodeObject:@(self.loadedFromCache) forKey:@"loadedFromCache"];
    [coder encodeObject:@(static_cast<unsigned char>(_info->resourceLoadType())) forKey:@"type"];
}

@end

