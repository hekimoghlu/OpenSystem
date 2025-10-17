/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#ifndef WKGeolocationPosition_h
#define WKGeolocationPosition_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKGeolocationPositionGetTypeID(void);

WK_EXPORT WKGeolocationPositionRef WKGeolocationPositionCreate(double timestamp, double latitude, double longitude, double accuracy);
WK_EXPORT WKGeolocationPositionRef WKGeolocationPositionCreate_b(double timestamp, double latitude, double longitude, double accuracy, bool providesAltitude, double altitude, bool providesAltitudeAccuracy, double altitudeAccuracy, bool providesHeading, double heading, bool providesSpeed, double speed);
WK_EXPORT WKGeolocationPositionRef WKGeolocationPositionCreate_c(double timestamp, double latitude, double longitude, double accuracy, bool providesAltitude, double altitude, bool providesAltitudeAccuracy, double altitudeAccuracy, bool providesHeading, double heading, bool providesSpeed, double speed, bool providesFloorLevel, double floorLevel);

#ifdef __cplusplus
}
#endif

#endif /* WKGeolocationPosition_h */
