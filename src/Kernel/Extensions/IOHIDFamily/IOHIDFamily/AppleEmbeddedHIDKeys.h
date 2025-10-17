/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#ifndef _IOKIT_HID_EMBEDDEDHIDKEYS_H_
#define _IOKIT_HID_EMBEDDEDHIDKEYS_H_

#include <sys/cdefs.h>

__BEGIN_DECLS

#define kIOHIDThresholdXKey                     "ThresholdX"
#define kIOHIDThresholdYKey                     "ThresholdY"
#define kIOHIDThresholdZKey                     "ThresholdZ"
#define kIOHIDThresholdPeriodKey                "ThresholdPeriod"


#define kIOHIDAccelerometerShakeKey             "Shake"
#define kIOHIDGyroShakeKey                      "Shake"

#define kIOHIDOrientationKey                    "Orientation"

/*!
 @typedef IOHIDOrientationType
 @abstract Orientation of event triggered.
 @discussion
 @constant kIOHIDOrientationTypeUndefined
 @constant kIOHIDOrientationTypeNorth
 @constant kIOHIDOrientationTypeSouth
 @constant kIOHIDOrientationTypeEast
 @constant kIOHIDOrientationTypeWest
 @constant kIOHIDOrientationTypeNorthEast
 @constant kIOHIDOrientationTypeNorthWest
 @constant kIOHIDOrientationTypeSoutEast
 @constant kIOHIDOrientationTypeSouthWest
 */
enum {
    kIOHIDOrientationTypeUndefined  = 0,
    kIOHIDOrientationTypeNorth      = 1,
    kIOHIDOrientationTypeSouth      = 2,
    kIOHIDOrientationTypeEast       = 3,
    kIOHIDOrientationTypeWest       = 4,
    kIOHIDOrientationTypeNorthEast  = 5,
    kIOHIDOrientationTypeNorthWest  = 6,
    kIOHIDOrientationTypeSouthEast  = 7,
    kIOHIDOrientationTypeSouthWest  = 8
};
typedef uint32_t IOHIDOrientationType;

#define kIOHIDPlacementKey                        "Placement"
/*!
 @typedef IOHIDPlacementType
 @abstract Placement of event triggered.
 @discussion
 @constant kIOHIDPlacementTypeUndefined
 @constant kIOHIDPlacementTypeTop
 @constant kIOHIDPlacementTypeBottom
 */
enum {
    kIOHIDPlacementTypeUndefined = 0,
    kIOHIDPlacementTypeTop       = 1,
    kIOHIDPlacementTypeBottom    = 2
};
typedef uint32_t IOHIDPlacementType;



__END_DECLS

#endif /* !_IOKIT_HID_EMBEDDEDHIDKEYS_H_ */
