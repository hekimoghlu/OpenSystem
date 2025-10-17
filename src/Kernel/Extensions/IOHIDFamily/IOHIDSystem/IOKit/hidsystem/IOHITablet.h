/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#ifndef _IOHITABLET_H
#define _IOHITABLET_H

#include <IOKit/hidsystem/IOHIPointing.h>
#include <IOKit/hidsystem/IOLLEvent.h>

class IOHITabletPointer;

#define kIOHIVendorID		"VendorID"
#define kIOHISystemTabletID	"SystemTabletID"
#define kIOHIVendorTabletID	"VendorTabletID"

typedef void (*TabletEventAction)(OSObject		*target,
                                  NXEventData	*tabletData,	// Do we want to parameterize this?
                                  AbsoluteTime ts);

typedef void (*ProximityEventAction)(OSObject		*target,
                                     NXEventData	*proximityData,	// or this?
                                     AbsoluteTime ts);

/* Event Callback Definitions */
typedef void (*TabletEventCallback)(
                        /* target */       OSObject *    target,
                        /* event */        NXEventData * tabletData,
                        /* atTime */       AbsoluteTime  ts,
                        /* sender */       OSObject *    sender,
                        /* refcon */       void *        refcon);

typedef void (*ProximityEventCallback)(
                        /* target */       OSObject *    target,
                        /* event */        NXEventData * proximityData,
                        /* atTime */       AbsoluteTime  ts,
                        /* sender */       OSObject *    sender,
                        /* refcon */       void *        refcon);

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHITablet : public IOHIPointing
#else
class IOHITablet : public IOHIPointing
#endif
{
    OSDeclareDefaultStructors(IOHITablet);
    friend class IOHITabletPointer;

public:
    UInt16		_systemTabletID;

private:
    OSObject *				_tabletEventTarget;
    TabletEventAction		_tabletEventAction;
    OSObject *				_proximityEventTarget;
    ProximityEventAction	_proximityEventAction;

protected:
    virtual void dispatchTabletEvent(NXEventData *tabletEvent,
                                     AbsoluteTime ts);

    virtual void dispatchProximityEvent(NXEventData *proximityEvent,
                                        AbsoluteTime ts);

    virtual bool startTabletPointer(IOHITabletPointer *pointer, OSDictionary *properties);

public:
    static UInt16 generateTabletID();

    virtual bool init(OSDictionary * propTable) APPLE_KEXT_OVERRIDE;
    virtual bool open(IOService *	client,
                      IOOptionBits	options,
                      RelativePointerEventAction	rpeAction,
                      AbsolutePointerEventAction	apeAction,
                      ScrollWheelEventAction		sweAction,
                      TabletEventAction			tabletAction,
                      ProximityEventAction		proximityAction);

    bool open(        IOService *			client,
                      IOOptionBits			options,
                      void *,
                      RelativePointerEventCallback	rpeCallback,
                      AbsolutePointerEventCallback	apeCallback,
                      ScrollWheelEventCallback		sweCallback,
                      TabletEventCallback		tabletCallback,
                      ProximityEventCallback		proximityCallback);
                      
private:

  static void _tabletEvent(IOHITablet *self,
                           NXEventData *tabletData,
                           AbsoluteTime ts);

  static void _proximityEvent(IOHITablet *self,
                              NXEventData *proximityData,
                              AbsoluteTime ts);


};

#endif /* !_IOHITABLET_H */
