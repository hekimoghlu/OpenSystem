/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#import "_WKGeolocationPositionInternal.h"

#if PLATFORM(IOS_FAMILY)

#import <CoreLocation/CLLocation.h>
#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKGeolocationPosition

+ (instancetype)positionWithLocation:(CLLocation *)location
{
    if (!location)
        return nil;

    return wrapper(WebKit::WebGeolocationPosition::create(WebCore::GeolocationPositionData { location })).autorelease();
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKGeolocationPosition.class, self))
        return;

    _geolocationPosition->~WebGeolocationPosition();

    [super dealloc];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_geolocationPosition;
}

@end

#endif // PLATFORM(IOS_FAMILY)
