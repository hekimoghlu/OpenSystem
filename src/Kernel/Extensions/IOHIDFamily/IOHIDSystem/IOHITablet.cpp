/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#include <AssertMacros.h>

#include "IOHITablet.h"
#include "IOHITabletPointer.h"

OSDefineMetaClassAndStructors(IOHITablet, IOHIPointing);

UInt16 IOHITablet::generateTabletID()
{
    static UInt16 _nextTabletID = 0x8000;
    return _nextTabletID++;
}

bool IOHITablet::init(OSDictionary *propTable)
{
    if (!IOHIPointing::init(propTable)) {
        return false;
    }

    _systemTabletID = 0;

    return true;
}

bool IOHITablet::open(IOService *			client,
                      IOOptionBits 			options,
                      RelativePointerEventAction	rpeAction,
                      AbsolutePointerEventAction	apeAction,
                      ScrollWheelEventAction		sweAction,
                      TabletEventAction			tabletAction,
                      ProximityEventAction		proximityAction)
{
    
    if (client == this) return true;
     
    return open(client, 
                options, 
                0,
                (RelativePointerEventCallback)rpeAction, 
                (AbsolutePointerEventCallback)apeAction, 
                (ScrollWheelEventCallback)sweAction, 
                (TabletEventCallback)tabletAction, 
                (ProximityEventCallback)proximityAction);
}

bool IOHITablet::open(IOService *			client,
                      IOOptionBits			options,
                      void *				/*refcon*/,
                      RelativePointerEventCallback	rpeCallback,
                      AbsolutePointerEventCallback	apeCallback,
                      ScrollWheelEventCallback		sweCallback,
                      TabletEventCallback		tabletCallback,
                      ProximityEventCallback		proximityCallback)
{
    if (client == this) return true;

    if (!IOHIPointing::open(client, 
                            options,
                            0,
                            rpeCallback, 
                            apeCallback, 
                            sweCallback)) {
        return false;
    }

    _tabletEventTarget = client;
    _tabletEventAction = (TabletEventAction)tabletCallback;
    _proximityEventTarget = client;
    _proximityEventAction = (ProximityEventAction)proximityCallback;

    return open(this, 
                options, 
                (RelativePointerEventAction)IOHIPointing::_relativePointerEvent, 
                (AbsolutePointerEventAction)IOHIPointing::_absolutePointerEvent, 
                (ScrollWheelEventAction)IOHIPointing::_scrollWheelEvent, 
                (TabletEventAction)_tabletEvent, 
                (ProximityEventAction)_proximityEvent);
}


void IOHITablet::dispatchTabletEvent(NXEventData *tabletEvent,
                                     AbsoluteTime ts)
{
    _tabletEvent(   this,
                    tabletEvent,
                    ts);
}

void IOHITablet::dispatchProximityEvent(NXEventData *proximityEvent,
                                        AbsoluteTime ts)
{
    _proximityEvent(this,
                    proximityEvent,
                    ts);
}

bool IOHITablet::startTabletPointer(IOHITabletPointer *pointer, OSDictionary *properties)
{
    require(pointer, no_attach);
    require(pointer->init(properties), no_attach);
    require(pointer->attach(this), no_attach);
    require(pointer->start(this), no_start);
    
no_start:
    pointer->detach(this);
no_attach:
    return false;
}

void IOHITablet::_tabletEvent(IOHITablet *self,
                           NXEventData *tabletData,
                           AbsoluteTime ts)
{
    TabletEventCallback teCallback;
    
    if (!(teCallback = (TabletEventCallback)self->_tabletEventAction) ||
        !tabletData)
        return;
        
    (*teCallback)(
                    self->_tabletEventTarget,
                    tabletData,
                    ts,
                    self,
                    0);
}

void IOHITablet::_proximityEvent(IOHITablet *self,
                              NXEventData *proximityData,
                              AbsoluteTime ts)
{
    ProximityEventCallback peCallback;
    
    if (!(peCallback = (ProximityEventCallback)self->_proximityEventAction) ||
        !proximityData)
        return;
            
    if (self->_systemTabletID == 0)
    {
        self->_systemTabletID = IOHITablet::generateTabletID();
        self->setProperty(kIOHISystemTabletID, (unsigned long long)self->_systemTabletID, 16);
    }

    proximityData->proximity.systemTabletID = self->_systemTabletID;

    (*peCallback)(  
                    self->_proximityEventTarget,
                    proximityData,
                    ts,
                    self,
                    0);
}



