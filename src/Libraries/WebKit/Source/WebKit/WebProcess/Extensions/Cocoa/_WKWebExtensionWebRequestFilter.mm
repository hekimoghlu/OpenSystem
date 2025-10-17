/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "_WKWebExtensionWebRequestFilter.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "ResourceLoadInfo.h"
#import "WKWebExtensionMatchPattern.h"
#import "WebExtensionTabIdentifier.h"
#import "WebExtensionUtilities.h"
#import "WebExtensionWindowIdentifier.h"
#import "_WKResourceLoadInfo.h"

using namespace WebKit;

static NSString *urlsKey = @"urls";
static NSString *typesKey = @"types";

static NSString *tabIdKey = @"tabId";
static NSString *windowIdKey = @"windowId";

_WKWebExtensionWebRequestResourceType toWebExtensionWebRequestResourceType(const ResourceLoadInfo& resourceLoadInfo)
{
    switch (resourceLoadInfo.type) {
    case ResourceLoadInfo::Type::Document:
        return resourceLoadInfo.parentFrameID ? _WKWebExtensionWebRequestResourceTypeMainFrame : _WKWebExtensionWebRequestResourceTypeSubframe;
    case ResourceLoadInfo::Type::Stylesheet:
        return _WKWebExtensionWebRequestResourceTypeStylesheet;
    case ResourceLoadInfo::Type::Script:
        return _WKWebExtensionWebRequestResourceTypeScript;
    case ResourceLoadInfo::Type::Image:
        return _WKWebExtensionWebRequestResourceTypeImage;
    case ResourceLoadInfo::Type::Font:
        return _WKWebExtensionWebRequestResourceTypeFont;
    case ResourceLoadInfo::Type::Object:
        return _WKWebExtensionWebRequestResourceTypeObject;
    case ResourceLoadInfo::Type::Fetch:
    case ResourceLoadInfo::Type::XMLHTTPRequest:
        return _WKWebExtensionWebRequestResourceTypeXMLHTTPRequest;
    case ResourceLoadInfo::Type::CSPReport:
        return _WKWebExtensionWebRequestResourceTypeCSPReport;
    case ResourceLoadInfo::Type::Media:
        return _WKWebExtensionWebRequestResourceTypeMedia;
    case ResourceLoadInfo::Type::ApplicationManifest:
        return _WKWebExtensionWebRequestResourceTypeApplicationManifest;
    case ResourceLoadInfo::Type::XSLT:
        return _WKWebExtensionWebRequestResourceTypeXSLT;
    case ResourceLoadInfo::Type::Ping:
        return _WKWebExtensionWebRequestResourceTypePing;
    case ResourceLoadInfo::Type::Beacon:
        return _WKWebExtensionWebRequestResourceTypeBeacon;
    case ResourceLoadInfo::Type::Other:
        return _WKWebExtensionWebRequestResourceTypeOther;
    }

    ASSERT_NOT_REACHED();
    return _WKWebExtensionWebRequestResourceTypeOther;
}

@implementation _WKWebExtensionWebRequestFilter {
    std::optional<WebExtensionTabIdentifier> _tabID;
    std::optional<WebExtensionWindowIdentifier> _windowID;
    NSArray<WKWebExtensionMatchPattern *> *_urlPatterns;
    NSSet<NSNumber *> *_types;
}

- (instancetype)initWithDictionary:(NSDictionary<NSString *, id> *)dictionary outErrorMessage:(NSString **)outErrorMessage
{
    if (!(self = [super init])) {
        *outErrorMessage = @"Runtime failure.";
        return nil;
    }

    static NSArray<NSString *> *requiredKeys = @[
        urlsKey,
    ];

    static NSDictionary<NSString *, id> *expectedTypes = @{
        urlsKey: @[ NSString.class ],
        typesKey: NSArray.class,
        tabIdKey: NSNumber.class,
        windowIdKey: NSNumber.class,
    };

    if (!validateDictionary(dictionary, nil, requiredKeys, expectedTypes, outErrorMessage))
        return nil;

    if ((*outErrorMessage = [self _initializeWithDictionary:dictionary]))
        return nil;

    return self;
}

static NSArray<WKWebExtensionMatchPattern *> *toMatchPatterns(NSArray<NSString *> *value, NSString **outErrorMessage)
{
    if (!value.count)
        return nil;

    NSError *error;

    NSMutableArray<WKWebExtensionMatchPattern *> *patterns = [[NSMutableArray alloc] init];
    for (NSString *rawPattern in value) {
        WKWebExtensionMatchPattern *pattern = [[WKWebExtensionMatchPattern alloc] initWithString:rawPattern error:&error];
        if (!pattern) {
            if (outErrorMessage)
                *outErrorMessage = toErrorString(nullString(), urlsKey, @"'%@' is an invalid match pattern. %@", rawPattern, error.localizedDescription);
            return nil;
        }

        [patterns addObject:pattern];
    }

    return patterns;
}

static NSNumber *toResourceType(NSString *typeString, NSString **outErrorMessage)
{
    static NSDictionary<NSString *, NSNumber *> *validTypes = @{
        @"main_frame": @(_WKWebExtensionWebRequestResourceTypeMainFrame),
        @"sub_frame": @(_WKWebExtensionWebRequestResourceTypeSubframe),
        @"stylesheet": @(_WKWebExtensionWebRequestResourceTypeStylesheet),
        @"script": @(_WKWebExtensionWebRequestResourceTypeScript),
        @"image": @(_WKWebExtensionWebRequestResourceTypeImage),
        @"font": @(_WKWebExtensionWebRequestResourceTypeFont),
        @"object": @(_WKWebExtensionWebRequestResourceTypeObject),
        @"xmlhttprequest": @(_WKWebExtensionWebRequestResourceTypeXMLHTTPRequest),
        @"ping": @(_WKWebExtensionWebRequestResourceTypePing),
        @"csp_report": @(_WKWebExtensionWebRequestResourceTypeCSPReport),
        @"media": @(_WKWebExtensionWebRequestResourceTypeMedia),
        @"websocket": @(_WKWebExtensionWebRequestResourceTypeWebsocket),
        @"web_manifest": @(_WKWebExtensionWebRequestResourceTypeApplicationManifest),
        @"xslt": @(_WKWebExtensionWebRequestResourceTypeXSLT),
        @"beacon": @(_WKWebExtensionWebRequestResourceTypeBeacon),
        @"other": @(_WKWebExtensionWebRequestResourceTypeOther),
    };

    NSNumber *typeAsNumber = validTypes[typeString];
    if (!typeAsNumber) {
        *outErrorMessage = toErrorString(nullString(), typesKey, @"'%@' is an unknown resource type", typeString);
        return nil;
    }

    return typeAsNumber;
}

static NSSet<NSNumber *> *toResourceTypes(NSArray<NSString *> *rawTypes, NSString **outErrorMessage)
{
    if (!rawTypes.count)
        return nil;

    NSMutableSet<NSNumber *> *types = [[NSMutableSet alloc] init];
    for (NSString *rawType in rawTypes) {
        NSNumber *type = toResourceType(rawType, outErrorMessage);
        if (!type)
            return nil;
        [types addObject:type];
    }

    return types;
}

static std::optional<WebExtensionTabIdentifier> toTabID(NSNumber *rawValue)
{
    if (!rawValue)
        return std::nullopt;

    return toWebExtensionTabIdentifier(rawValue.doubleValue);
}

static std::optional<WebExtensionWindowIdentifier> toWindowID(NSNumber *rawValue)
{
    if (!rawValue)
        return std::nullopt;

    return toWebExtensionWindowIdentifier(rawValue.doubleValue);
}

- (NSString *)_initializeWithDictionary:(NSDictionary<NSString *, id> *)dictionary
{
    NSString *errorMessage;
    _urlPatterns = toMatchPatterns(dictionary[urlsKey], &errorMessage);
    if (errorMessage)
        return errorMessage;

    _types = toResourceTypes(dictionary[typesKey], &errorMessage);
    if (errorMessage)
        return errorMessage;

    _tabID = toTabID(dictionary[tabIdKey]);
    _windowID = toWindowID(dictionary[windowIdKey]);

    return nil;
}

- (BOOL)matchesRequestForResourceOfType:(_WKWebExtensionWebRequestResourceType)resourceType URL:(NSURL *)URL tabID:(double)tabID windowID:(double)windowID
{
    if (_types && ![_types containsObject:@(resourceType)])
        return NO;

    if (_urlPatterns) {
        BOOL hasURLMatch = NO;
        for (WKWebExtensionMatchPattern *pattern in _urlPatterns) {
            if ([pattern matchesURL:URL]) {
                hasURLMatch = YES;
                break;
            }
        }

        if (!hasURLMatch)
            return NO;
    }

    if (isValid(_tabID) && toWebAPI(_tabID.value()) != tabID)
        return NO;

    if (isValid(_windowID) && toWebAPI(_windowID.value()) != windowID)
        return NO;

    return YES;
}

@end

#else

@implementation _WKWebExtensionWebRequestFilter

- (instancetype)initWithDictionary:(NSDictionary<NSString *, id> *)dictionary outErrorMessage:(NSString **)outErrorMessage
{
    return nil;
}

- (BOOL)matchesRequestForResourceOfType:(_WKWebExtensionWebRequestResourceType)resourceType URL:(NSURL *)URL tabID:(double)tabID windowID:(double)windowID
{
    return NO;
}

_WKWebExtensionWebRequestResourceType toWebExtensionWebRequestResourceType(const WebKit::ResourceLoadInfo&)
{
    return _WKWebExtensionWebRequestResourceTypeOther;
}

@end

#endif // ENABLE(WEB_EXTENSIONS)
