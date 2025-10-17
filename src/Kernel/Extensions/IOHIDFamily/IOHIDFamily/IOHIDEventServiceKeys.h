/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#ifndef IOHIDEventServiceKeys_h
#define IOHIDEventServiceKeys_h

/*!
 * @define kIOHIDPointerAccelerationKey
 *
 * @abstract
 * Number property that contains the pointer acceleration value.
 */
#define kIOHIDPointerAccelerationKey "HIDPointerAcceleration"

#define kIOHIDTrackpadScrollAccelerationKey "HIDTrackpadScrollAcceleration"

#define kIOHIDTrackpadAccelerationType  "HIDTrackpadAcceleration"

/*!
 * @define kIOHIDPointerAccelerationTypeKey
 *
 * @abstract
 * String property containing the type of acceleration for pointer.
 * Supported types are:
 *      <code>kIOHIDPointerAccelerationKey</code>
 *      <code>kIOHIDMouseScrollAccelerationKey</code>
 *      <code>kIOHIDTrackpadAccelerationType</code>
 */
#define kIOHIDPointerAccelerationTypeKey "HIDPointerAccelerationType"

/*!
 * @define kIOHIDMouseScrollAccelerationKey
 *
 * @abstract
 * Number property that contains the mouse scroll acceleration value.
 */
#define kIOHIDMouseScrollAccelerationKey "HIDMouseScrollAcceleration"

/*!
 * @define kIOHIDMouseAccelerationTypeKey
 *
 * @abstract
 * Number property that contains the mouse acceleration value.
 */
#define kIOHIDMouseAccelerationTypeKey "HIDMouseAcceleration"

/*!
 * @define kIOHIDScrollAccelerationKey
 *
 * @abstract
 * Number property that contains the scroll acceleration value.
 */
#define kIOHIDScrollAccelerationKey "HIDScrollAcceleration"

/*!
 * @define kIOHIDScrollAccelerationTypeKey
 *
 * @abstract
 * Number property containing the type of acceleration for scroll.
 * Supported types are:
 *      <code>kIOHIDMouseScrollAccelerationKey</code>
 *      <code>kIOHIDTrackpadScrollAccelerationKey</code>
 */
#define kIOHIDScrollAccelerationTypeKey "HIDScrollAccelerationType"

/*!
 * @define kIOHIDDigitizerTipThresholdKey
 *
 * @abstract
 * Number property that describes the threshold percentage for when the tip
 * pressure of a digitizer stylus should change from hovering to dragging.
 *
 * @discussion
 * If a digitizer stylus supports the kHIDUsage_Dig_TipPressure (0x30) usage,
 * the service may optionally publish this key to describe the value at which
 * the pressure should change the pointer behavior from hovering to dragging.
 * The value is a percentage from 0 to 100, where 100 percent is equal to the
 * logical max that the stylus dispatches. If no value is provided, the default
 * value of 75 will be used.
 */
#define kIOHIDDigitizerTipThresholdKey "DigitizerTipThreshold"

/*!
 * @define kIOHIDSurfaceDimensionsKey
 *
 * @abstract
 * Dictionary property published on a service that describes the surface
 * dimensions for services that publish absolute X/Y values, such as digitizer
 * and pointer devices. The dictionary will contain the kIOHIDWidthKey and
 * kIOHIDHeightKey keys described below. Value is in millimeter represented
 * as IOFixed.
 */
#define kIOHIDSurfaceDimensionsKey "SurfaceDimensions"

/*!
 * @define kIOHIDWidthKey
 *
 * @abstract
 * Number property used in the surface dimensions dictionary described above.
 * Default value represents the physical max - physical min of the absolute
 * X value.
 */
#define kIOHIDWidthKey "Width"

/*!
 * @define kIOHIDHeightKey
 *
 * @abstract
 * Number property used in the surface dimensions dictionary described above.
 * Default value represents the physical max - physical min of the absolute
 * Y value.
 */
#define kIOHIDHeightKey "Height"

/*!
 * @define kIOHIDEventDriverHandlesReport
 *
 * @abstract
 * Boolean property used to let handleReport in an IOUserHIDEventDriver get the
 * report before any other processing is done in IOUserHIDEventService.
 * If this property is enabled the IOUserHIDEventService subclass should update
 * the elements with IOHIDInterface::processReport to update the IOHIDElements as
 * IOUserHIDEventService will not do this like when this property is not set.
 */
#define kIOHIDEventDriverHandlesReport "IOHIDEventDriverHandlesReport"


/*!
 * @define kIOHIDServiceAccelerationProperties
 *
 * @abstract Key for the properties to set by the IOHIDEventSystem to tune the
 *           HID Event System PointerScrollFilter parameters. The following keys
 *           are the supported parameters to tune for Acceleration.
 */
#define kIOHIDServiceAccelerationProperties "IOHIDSetAcceleration"

#define kIOHIDPointerAccelerationMultiplierKey        "HIDPointerAccelerationMultiplier"

#define kHIDPointerReportRateKey                "HIDPointerReportRate"

#define kIOHIDScrollReportRateKey       "HIDScrollReportRate"

#define kHIDAccelParametricCurvesKey            "HIDAccelCurves"

#define kHIDScrollAccelParametricCurvesKey      "HIDScrollAccelCurves"

#define kIOHIDScrollResolutionKey       "HIDScrollResolution"

#define kIOHIDDropAccelPropertyEventsKey "DropAccelPropertyEvents"

#define kIOHIDScrollResolutionXKey      "HIDScrollResolutionX"
#define kIOHIDScrollResolutionYKey      "HIDScrollResolutionY"
#define kIOHIDScrollResolutionZKey      "HIDScrollResolutionZ"

/*!
 @defined    kIOHIDUseLinearPointerAccelerationKey
 @abstract   Property to force use of linear scaling for mouse accleration.
 @discussion WindowServer sets this to true when the user wants linear tracking. Only has an effect if this device uses HIDMouseAcceleration style acceleration.
 */
#define kIOHIDUseLinearScalingMouseAccelerationKey "HIDUseLinearScalingMouseAcceleration"

/*!
     @defined    kIOHIDPointerAccelerationSupportKey
     @abstract   Property to turn enable/disable acceleration of relative pointer events
     @discussion A boolean value to enable devices that report movement precisely but using relative positions,
                    if false the events from the device will not have acceleration applied to the event value calculation.
                    If the key is not set then the device will have acceleration applied to it's events by default.
 */
#define kIOHIDPointerAccelerationSupportKey    "HIDSupportsPointerAcceleration"

/*!
     @defined    kIOHIDScrollAccelerationSupportKey
     @abstract   Property to turn enable/disable acceleration of scroll events
     @discussion A boolean value to enable devices that report scroll precisely but using relative positions,
                    if false the events from the device will not have acceleration applied to the event value calculation.
                    If the key is not set then the device will have acceleration applied to it's events by default.
 */
#define kIOHIDScrollAccelerationSupportKey     "HIDSupportsScrollAcceleration"

#endif /* IOHIDDeviceTypes_h */
