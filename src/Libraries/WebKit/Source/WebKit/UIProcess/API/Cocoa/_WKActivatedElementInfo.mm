/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#import "_WKActivatedElementInfoInternal.h"

#import "CocoaImage.h"
#import <WebCore/ShareableBitmap.h>
#import <wtf/RetainPtr.h>

#if USE(APPKIT)
#import <AppKit/NSImage.h>
#else
#import <UIKit/UIImage.h>
#endif

@implementation _WKActivatedElementInfo  {
    RetainPtr<NSURL> _URL;
    RetainPtr<NSURL> _imageURL;
    RetainPtr<NSString> _title;
    WebCore::IntPoint _interactionLocation;
    RetainPtr<NSString> _ID;
    RefPtr<WebCore::ShareableBitmap> _image;
    RetainPtr<NSString> _imageMIMEType;
    RetainPtr<CocoaImage> _cocoaImage;
#if PLATFORM(IOS_FAMILY)
    RetainPtr<NSDictionary> _userInfo;
    BOOL _isAnimating;
    BOOL _canShowAnimationControls;
    Vector<WebCore::ElementAnimationContext> _animationsUnderElement;
#endif
    BOOL _animatedImage;
    BOOL _isImage;
    BOOL _isUsingAlternateURLForImage;
#if ENABLE(SPATIAL_IMAGE_DETECTION)
    BOOL _isSpatialImage;
#endif
}

#if PLATFORM(IOS_FAMILY)
+ (instancetype)activatedElementInfoWithInteractionInformationAtPosition:(const WebKit::InteractionInformationAtPosition&)information userInfo:(NSDictionary *)userInfo
{
    return adoptNS([[self alloc] _initWithInteractionInformationAtPosition:information isUsingAlternateURLForImage:NO userInfo:userInfo]).autorelease();
}

- (instancetype)_initWithInteractionInformationAtPosition:(const WebKit::InteractionInformationAtPosition&)information isUsingAlternateURLForImage:(BOOL)isUsingAlternateURLForImage userInfo:(NSDictionary *)userInfo
{
    if (!(self = [super init]))
        return nil;
    
    _URL = information.url;
    _imageURL = information.imageURL;
    _imageMIMEType = information.imageMIMEType;
    _interactionLocation = information.request.point;
    _title = information.title;
    _boundingRect = information.bounds;
    
    if (information.isAttachment)
        _type = _WKActivatedElementTypeAttachment;
    else if (information.isLink)
        _type = _WKActivatedElementTypeLink;
    else if (information.isImage)
        _type = _WKActivatedElementTypeImage;
    else
        _type = _WKActivatedElementTypeUnspecified;
    
    _image = information.image;
    _ID = information.idAttribute;
    _animatedImage = information.isAnimatedImage;
    _isAnimating = information.isAnimating;
    _canShowAnimationControls = information.canShowAnimationControls;
    _isImage = information.isImage;
    _isUsingAlternateURLForImage = isUsingAlternateURLForImage;
#if ENABLE(SPATIAL_IMAGE_DETECTION)
    _isSpatialImage = information.isSpatialImage;
#endif
    _userInfo = userInfo;
#if ENABLE(ACCESSIBILITY_ANIMATION_CONTROL)
    _animationsUnderElement = information.animationsAtPoint;
#endif

    return self;
}

- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url information:(const WebKit::InteractionInformationAtPosition&)information
{
    return [self _initWithType:type URL:url imageURL:information.imageURL information:information];
}

- (instancetype)_initWithType:(_WKActivatedElementType)type image:(WebCore::ShareableBitmap*)image information:(const WebKit::InteractionInformationAtPosition&)information
{
    return [self _initWithType:type URL:information.url imageURL:information.imageURL image:image userInfo:nil information:information];
}

- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url imageURL:(NSURL *)imageURL information:(const WebKit::InteractionInformationAtPosition&)information
{
    return [self _initWithType:type URL:url imageURL:imageURL image:information.image.get() userInfo:nil information:information];
}

- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url image:(WebCore::ShareableBitmap*)image information:(const WebKit::InteractionInformationAtPosition&)information
{
    return [self _initWithType:type URL:url imageURL:information.imageURL image:image userInfo:nil information:information];
}

- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url imageURL:(NSURL *)imageURL userInfo:(NSDictionary *)userInfo information:(const WebKit::InteractionInformationAtPosition&)information
{
    return [self _initWithType:type URL:url imageURL:imageURL image:information.image.get() userInfo:userInfo information:information];
}

- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url imageURL:(NSURL *)imageURL image:(WebCore::ShareableBitmap*)image userInfo:(NSDictionary *)userInfo information:(const WebKit::InteractionInformationAtPosition&)information
{
#if ENABLE(ACCESSIBILITY_ANIMATION_CONTROL)
    auto animationsAtPoint = information.animationsAtPoint;
#else
    Vector<WebCore::ElementAnimationContext> animationsAtPoint;
#endif

    return [self _initWithType:type URL:url imageURL:imageURL location:information.request.point title:information.title ID:information.idAttribute rect:information.bounds image:image imageMIMEType:information.imageMIMEType isAnimatedImage:information.isAnimatedImage isAnimating:information.isAnimating canShowAnimationControls:information.canShowAnimationControls animationsUnderElement:animationsAtPoint userInfo:userInfo];
}
#endif // PLATFORM(IOS_FAMILY)

- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url imageURL:(NSURL *)imageURL location:(const WebCore::IntPoint&)location title:(NSString *)title ID:(NSString *)ID rect:(CGRect)rect image:(WebCore::ShareableBitmap*)image imageMIMEType:(NSString *)imageMIMEType isAnimatedImage:(BOOL)isAnimatedImage isAnimating:(BOOL)isAnimating canShowAnimationControls:(BOOL)canShowAnimationControls animationsUnderElement:(Vector<WebCore::ElementAnimationContext>)animationsUnderElement userInfo:(NSDictionary *)userInfo
{
    if (!(self = [super init]))
        return nil;

    _URL = adoptNS([url copy]);
    _imageURL = adoptNS([imageURL copy]);
    _imageMIMEType = adoptNS(imageMIMEType.copy);
    _interactionLocation = location;
    _title = adoptNS([title copy]);
    _boundingRect = rect;
    _type = type;
    _image = image;
    _ID = ID;
#if PLATFORM(IOS_FAMILY)
    _userInfo = adoptNS([userInfo copy]);
    _isAnimating = isAnimating;
    _canShowAnimationControls = canShowAnimationControls;
    _animationsUnderElement = animationsUnderElement;
#endif
    _animatedImage = isAnimatedImage;

    return self;
}

- (NSURL *)URL
{
    return _URL.get();
}

- (NSURL *)imageURL
{
    return _imageURL.get();
}

- (NSString *)title
{
    return _title.get();
}

- (NSString *)imageMIMEType
{
    return _imageMIMEType.get();
}

- (NSString *)ID
{
    return _ID.get();
}

- (WebCore::IntPoint)_interactionLocation
{
    return _interactionLocation;
}

- (BOOL)isAnimatedImage
{
    return _animatedImage;
}

- (BOOL)_isUsingAlternateURLForImage
{
    return _isUsingAlternateURLForImage;
}

#if ENABLE(SPATIAL_IMAGE_DETECTION)
- (BOOL)isSpatialImage
{
    return _isSpatialImage;
}
#endif

#if PLATFORM(IOS_FAMILY)
- (BOOL)isAnimating
{
    return _isAnimating;
}

- (BOOL)canShowAnimationControls
{
    return _canShowAnimationControls;
}

- (const Vector<WebCore::ElementAnimationContext>&)_animationsUnderElement
{
    return _animationsUnderElement;
}

- (NSDictionary *)userInfo
{
    return _userInfo.get();
}
#endif

- (BOOL)_isImage
{
    return _isImage;
}

- (CocoaImage *)image
{
    if (_cocoaImage)
        return adoptNS([_cocoaImage copy]).autorelease();

    if (!_image)
        return nil;

#if USE(APPKIT)
    _cocoaImage = adoptNS([[NSImage alloc] initWithCGImage:_image->makeCGImageCopy().get() size:NSSizeFromCGSize(_boundingRect.size)]);
#else
    _cocoaImage = adoptNS([[UIImage alloc] initWithCGImage:_image->makeCGImageCopy().get()]);
#endif
    _image = nullptr;

    return adoptNS([_cocoaImage copy]).autorelease();
}

@end
