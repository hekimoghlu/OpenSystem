/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#if PLATFORM(IOS_FAMILY)
#import "InteractionInformationAtPosition.h"
#endif
#import <WebCore/ElementAnimationContext.h>
#import <WebCore/IntPoint.h>
#import <WebKit/_WKActivatedElementInfo.h>

namespace WebCore {
class ShareableBitmap;
}

@interface _WKActivatedElementInfo ()

#if PLATFORM(IOS_FAMILY)
+ (instancetype)activatedElementInfoWithInteractionInformationAtPosition:(const WebKit::InteractionInformationAtPosition&)information userInfo:(NSDictionary *)userInfo;
- (instancetype)_initWithInteractionInformationAtPosition:(const WebKit::InteractionInformationAtPosition&)information isUsingAlternateURLForImage:(BOOL)isUsingAlternateURLForImage userInfo:(NSDictionary *)userInfo;
- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url information:(const WebKit::InteractionInformationAtPosition&)information;
- (instancetype)_initWithType:(_WKActivatedElementType)type image:(WebCore::ShareableBitmap*)image information:(const WebKit::InteractionInformationAtPosition&)information;
- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url imageURL:(NSURL *)imageURL information:(const WebKit::InteractionInformationAtPosition&)information;
- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url image:(WebCore::ShareableBitmap*)image information:(const WebKit::InteractionInformationAtPosition&)information;
- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url imageURL:(NSURL *)imageURL userInfo:(NSDictionary *)userInfo information:(const WebKit::InteractionInformationAtPosition&)information;
- (instancetype)_initWithType:(_WKActivatedElementType)type URL:(NSURL *)url imageURL:(NSURL *)imageURL location:(const WebCore::IntPoint&)location title:(NSString *)title ID:(NSString *)ID rect:(CGRect)rect image:(WebCore::ShareableBitmap*)image imageMIMEType:(NSString *)imageMIMEType isAnimatedImage:(BOOL)isAnimatedImage isAnimating:(BOOL)isAnimating canShowAnimationControls:(BOOL)canShowAnimationControls animationsUnderElement:(Vector<WebCore::ElementAnimationContext>)animationsUnderElement userInfo:(NSDictionary *)userInfo;
#endif // PLATFORM(IOS_FAMILY)

@property (nonatomic, readonly) NSString *imageMIMEType;
@property (nonatomic, readonly) WebCore::IntPoint _interactionLocation;
@property (nonatomic, readonly) BOOL _isImage;
@property (nonatomic, readonly) BOOL _isUsingAlternateURLForImage;
@property (nonatomic, readonly) const Vector<WebCore::ElementAnimationContext>& _animationsUnderElement;

@end
