/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#ifndef _IOHIDPOINTING_H
#define _IOHIDPOINTING_H

#include <IOKit/hidsystem/IOHIDTypes.h>
#include "IOHITablet.h"
#include "IOHIDEventService.h"
#include "IOHIDevicePrivateKeys.h"

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHIDPointing : public IOHITablet
#else
class IOHIDPointing : public IOHITablet
#endif
{
    OSDeclareDefaultStructors(IOHIDPointing);

private:
    IOHIDEventService *      _provider;

    IOItemCount             _numButtons;
    IOFixed                 _resolution;
    IOFixed                 _scrollResolution;
    bool                    _isDispatcher;

public:
    static UInt16           generateDeviceID();

    // Allocator
    static IOHIDPointing * Pointing(
                                UInt32          buttonCount,
                                IOFixed         pointerResolution,
                                IOFixed         scrollResolution,
                                bool            isDispatcher);

    virtual bool initWithMouseProperties(
                                UInt32          buttonCount,
                                IOFixed         pointerResolution,
                                IOFixed         scrollResolution,
                                bool            isDispatcher);

    virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;
    virtual void stop(IOService * provider) APPLE_KEXT_OVERRIDE;


    virtual void dispatchAbsolutePointerEvent(
                                AbsoluteTime                timeStamp,
                                IOGPoint *                  newLoc,
                                IOGBounds *                 bounds,
                                UInt32                      buttonState,
                                bool                        inRange,
                                SInt32                      tipPressure,
                                SInt32                      tipPressureMin,
                                SInt32                      tipPressureMax,
                                IOOptionBits                options = 0);

	virtual void dispatchRelativePointerEvent(
                                AbsoluteTime                timeStamp,
								SInt32                      dx,
								SInt32                      dy,
								UInt32                      buttonState,
								IOOptionBits                options = 0);

	virtual void dispatchScrollWheelEvent(
                                AbsoluteTime                timeStamp,
								SInt32                      deltaAxis1,
								SInt32                      deltaAxis2,
								UInt32                      deltaAxis3,
								IOOptionBits                options = 0);

    virtual void dispatchTabletEvent(
                                    NXEventData *           tabletEvent,
                                    AbsoluteTime            ts) APPLE_KEXT_OVERRIDE;

    virtual void dispatchProximityEvent(
                                    NXEventData *           proximityEvent,
                                    AbsoluteTime            ts) APPLE_KEXT_OVERRIDE;

protected:
  virtual IOItemCount buttonCount(void) APPLE_KEXT_OVERRIDE;
  virtual IOFixed     resolution(void) APPLE_KEXT_OVERRIDE;

private:
  // This is needed to pass properties defined
  // in IOHIDDevice to the nub layer
  void	  setupProperties();

};

#endif /* !_IOHIDPOINTING_H */
