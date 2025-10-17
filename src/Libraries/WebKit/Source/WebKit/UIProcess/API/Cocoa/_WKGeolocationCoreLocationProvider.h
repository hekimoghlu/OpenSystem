/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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
#import <WebKit/WKFoundation.h>

#if TARGET_OS_IPHONE

#import <Foundation/NSObject.h>

NS_ASSUME_NONNULL_BEGIN

@class _WKGeolocationPosition;

WK_API_AVAILABLE(macos(10.13), ios(11.0))
@protocol _WKGeolocationCoreLocationListener <NSObject>
- (void)geolocationAuthorizationGranted;
- (void)geolocationAuthorizationDenied;
- (void)positionChanged:(_WKGeolocationPosition *)position;
- (void)errorOccurred:(NSString *)errorMessage;
- (void)resetGeolocation;
@end

WK_API_AVAILABLE(macos(10.13), ios(11.0))
@protocol _WKGeolocationCoreLocationProvider <NSObject>
- (void)setListener:(id <_WKGeolocationCoreLocationListener>)listener;
- (void)requestGeolocationAuthorization;
- (void)start;
- (void)stop;
- (void)setEnableHighAccuracy:(BOOL)flag;
@end

NS_ASSUME_NONNULL_END

#endif // TARGET_OS_IPHONE
