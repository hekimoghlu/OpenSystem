/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#import "_WKHitTestResultInternal.h"

#import "WKFrameInfoInternal.h"
#import "WebPageProxy.h"

#if PLATFORM(MAC) || HAVE(UIKIT_WITH_MOUSE_SUPPORT)

#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKHitTestResult

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKHitTestResult.class, self))
        return;

    _hitTestResult->~HitTestResult();

    [super dealloc];
}

static NSURL *URLFromString(const WTF::String& urlString)
{
    return urlString.isEmpty() ? nil : [NSURL URLWithString:urlString];
}

- (NSURL *)absoluteImageURL
{
    return URLFromString(_hitTestResult->absoluteImageURL());
}

- (NSURL *)absolutePDFURL
{
    return URLFromString(_hitTestResult->absolutePDFURL());
}

- (NSURL *)absoluteLinkURL
{
    return URLFromString(_hitTestResult->absoluteLinkURL());
}

- (BOOL)hasLocalDataForLinkURL
{
    return _hitTestResult->hasLocalDataForLinkURL();
}

- (NSString *)linkLocalDataMIMEType
{
    return _hitTestResult->linkLocalDataMIMEType();
}

- (NSURL *)absoluteMediaURL
{
    return URLFromString(_hitTestResult->absoluteMediaURL());
}

- (NSString *)linkLabel
{
    return _hitTestResult->linkLabel();
}

- (NSString *)linkTitle
{
    return _hitTestResult->linkTitle();
}

- (NSString *)lookupText
{
    return _hitTestResult->lookupText();
}

- (NSString *)linkSuggestedFilename
{
    return _hitTestResult->linkSuggestedFilename();
}

- (NSString *)imageSuggestedFilename
{
    return _hitTestResult->imageSuggestedFilename();
}

- (NSString *)imageMIMEType
{
    return _hitTestResult->sourceImageMIMEType();
}

- (BOOL)isContentEditable
{
    return _hitTestResult->isContentEditable();
}

- (BOOL)isSelected
{
    return _hitTestResult->isSelected();
}

- (BOOL)isMediaDownloadable
{
    return _hitTestResult->isDownloadableMedia();
}

- (BOOL)isMediaFullscreen
{
    return _hitTestResult->mediaIsInFullscreen();
}

- (CGRect)elementBoundingBox
{
    return _hitTestResult->elementBoundingBox();
}

- (_WKHitTestResultElementType)elementType
{
    switch (_hitTestResult->elementType()) {
    case WebKit::WebHitTestResultData::ElementType::None:
        return _WKHitTestResultElementTypeNone;
    case WebKit::WebHitTestResultData::ElementType::Audio:
        return _WKHitTestResultElementTypeAudio;
    case WebKit::WebHitTestResultData::ElementType::Video:
        return _WKHitTestResultElementTypeVideo;
    }

    ASSERT_NOT_REACHED();
    return _WKHitTestResultElementTypeNone;
}

- (WKFrameInfo *)frameInfo
{
    if (auto frameInfo = _hitTestResult->frameInfo())
        return wrapper(API::FrameInfo::create(WTFMove(*frameInfo), _hitTestResult->page())).autorelease();
    return nil;
}

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_hitTestResult;
}

@end

#endif
