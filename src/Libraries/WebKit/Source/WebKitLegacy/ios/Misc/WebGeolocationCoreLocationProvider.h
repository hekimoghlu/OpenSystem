/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#if defined(__cplusplus)

#import <Foundation/Foundation.h>

namespace WebCore {
class GeolocationPositionData;
}

// WebGeolocationCoreLocationDelegate abstracts the location services of CoreLocation.
// All the results come back through the protocol GeolocationUpdateListener. Those callback can
// be done synchronously and asynchronously in responses to calls made on WebGeolocationCoreLocationDelegate.

// All calls to WebGeolocationCoreLocationDelegate must be on the main thread, all callbacks are done on the
// main thread.

@protocol WebGeolocationCoreLocationUpdateListener
- (void)geolocationAuthorizationGranted;
- (void)geolocationAuthorizationDenied;

- (void)positionChanged:(WebCore::GeolocationPositionData&&)position;
- (void)errorOccurred:(NSString *)errorMessage;
- (void)resetGeolocation;
@end


@interface WebGeolocationCoreLocationProvider : NSObject
- (id)initWithListener:(id<WebGeolocationCoreLocationUpdateListener>)listener;

- (void)requestGeolocationAuthorization;

- (void)start;
- (void)stop;

- (void)setEnableHighAccuracy:(BOOL)flag;
@end

#endif
