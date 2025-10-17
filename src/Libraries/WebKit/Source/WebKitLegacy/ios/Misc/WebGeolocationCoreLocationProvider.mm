/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

#import "WebGeolocationCoreLocationProvider.h"

#import <CoreLocation/CLLocation.h>
#import <CoreLocation/CLLocationManagerDelegate.h>
#import <CoreLocation/CoreLocation.h>
#import <WebCore/GeolocationPositionData.h>
#import <WebKitLogging.h>
#import <objc/objc-runtime.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK(CoreLocation)

SOFT_LINK_CLASS(CoreLocation, CLLocationManager)
SOFT_LINK_CLASS(CoreLocation, CLLocation)

SOFT_LINK_CONSTANT(CoreLocation, kCLLocationAccuracyBest, double)
SOFT_LINK_CONSTANT(CoreLocation, kCLLocationAccuracyHundredMeters, double)

#define kCLLocationAccuracyBest getkCLLocationAccuracyBest()
#define kCLLocationAccuracyHundredMeters getkCLLocationAccuracyHundredMeters()

using namespace WebCore;

@interface WebGeolocationCoreLocationProvider () <CLLocationManagerDelegate>
@end

@implementation WebGeolocationCoreLocationProvider
{
    id<WebGeolocationCoreLocationUpdateListener> _positionListener;
    RetainPtr<CLLocationManager> _locationManager;
    BOOL _isWaitingForAuthorization;
    CLAuthorizationStatus _lastAuthorizationStatus;
}

- (void)createLocationManager
{
    ASSERT(!_locationManager);

    _locationManager = adoptNS([allocCLLocationManagerInstance() init]);
    _lastAuthorizationStatus = [getCLLocationManagerClass() authorizationStatus];

    [ _locationManager setDelegate:self];
}

- (id)initWithListener:(id<WebGeolocationCoreLocationUpdateListener>)listener
{
    self = [super init];
    if (self) {
        _positionListener = listener;
        [self createLocationManager];
    }
    return self;
}

- (void)dealloc
{
    [_locationManager setDelegate:nil];
    [super dealloc];
}

- (void)requestGeolocationAuthorization
{
    if (![getCLLocationManagerClass() locationServicesEnabled]) {
        [_positionListener geolocationAuthorizationDenied];
        return;
    }

    switch ([getCLLocationManagerClass() authorizationStatus]) {
    case kCLAuthorizationStatusNotDetermined: {
        if (!_isWaitingForAuthorization) {
            _isWaitingForAuthorization = YES;
            [_locationManager requestWhenInUseAuthorization];
        }
        break;
    }
    case kCLAuthorizationStatusAuthorizedAlways:
    case kCLAuthorizationStatusAuthorizedWhenInUse: {
        [_positionListener geolocationAuthorizationGranted];
        break;
    }
    case kCLAuthorizationStatusRestricted:
    case kCLAuthorizationStatusDenied:
        [_positionListener geolocationAuthorizationDenied];
        break;
    }
}

static bool isAuthorizationGranted(CLAuthorizationStatus authorizationStatus)
{
    return authorizationStatus == kCLAuthorizationStatusAuthorizedAlways || authorizationStatus == kCLAuthorizationStatusAuthorizedWhenInUse;
}

- (void)start
{
    if (![getCLLocationManagerClass() locationServicesEnabled]
        || !isAuthorizationGranted([getCLLocationManagerClass() authorizationStatus])) {
        [_locationManager stopUpdatingLocation];
        [_positionListener resetGeolocation];
        return;
    }

    [_locationManager startUpdatingLocation];
}

- (void)stop
{
    [_locationManager stopUpdatingLocation];
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (void)locationManager:(CLLocationManager *)manager didChangeAuthorizationStatus:(CLAuthorizationStatus)status
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    if (_isWaitingForAuthorization) {
        switch (status) {
        case kCLAuthorizationStatusNotDetermined:
            // This can happen after resume if the user has still not answered the dialog. We just have to wait for the permission.
            break;
        case kCLAuthorizationStatusDenied:
        case kCLAuthorizationStatusRestricted:
            _isWaitingForAuthorization = NO;
            [_positionListener geolocationAuthorizationDenied];
            break;
        case kCLAuthorizationStatusAuthorizedAlways:
        case kCLAuthorizationStatusAuthorizedWhenInUse:
            _isWaitingForAuthorization = NO;
            [_positionListener geolocationAuthorizationGranted];
            break;
        }
    } else {
        if (!(isAuthorizationGranted(_lastAuthorizationStatus) && isAuthorizationGranted(status))) {
            [_locationManager stopUpdatingLocation];
            [_positionListener resetGeolocation];
        }
    }
    _lastAuthorizationStatus = status;
}

- (void)sendLocation:(CLLocation *)newLocation
{
    [_positionListener positionChanged:GeolocationPositionData { newLocation }];
}

- (void)locationManager:(CLLocationManager *)manager didUpdateLocations:(NSArray *)locations
{
    UNUSED_PARAM(manager);
    for (CLLocation *location in locations)
        [self sendLocation:location];
}

- (void)locationManager:(CLLocationManager *)manager didFailWithError:(NSError *)error
{
    ASSERT(_positionListener);
    ASSERT(error);
    UNUSED_PARAM(manager);

    if ([error code] == kCLErrorDenied) {
        // Ignore the error here and let locationManager:didChangeAuthorizationStatus: handle the permission.
        return;
    }

    NSString *errorMessage = [error localizedDescription];
    [_positionListener errorOccurred:errorMessage];
}

- (void)setEnableHighAccuracy:(BOOL)flag
{
    [_locationManager setDesiredAccuracy:flag ? kCLLocationAccuracyBest : kCLLocationAccuracyHundredMeters];
}

@end

#endif // PLATFORM(IOS_FAMILY)
