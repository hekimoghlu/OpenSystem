/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#ifndef _APPLEUSERHIDEVENTSERVICE_H
#define _APPLEUSERHIDEVENTSERVICE_H

#include <IOKit/hidevent/IOHIDEventService.h>
#include <IOKit/hidevent/IOHIDEventDriver.h>
#include <sys/queue.h>

struct EventCopyCaller;
struct SetPropertiesCaller;
struct SetLEDCaller;

class AppleUserHIDEventService: public IOHIDEventDriver
{
    OSDeclareDefaultStructors (AppleUserHIDEventService)

private:    
  
    STAILQ_HEAD(EventCopyCallerList, EventCopyCaller);
    STAILQ_HEAD(SetPropertiesCallerList, SetPropertiesCaller);
    STAILQ_HEAD(SetLEDCallerList, SetLEDCaller);

    struct AppleUserHIDEventService_IVars
    {
        OSArray               * elements;
        IOHIDInterface        * provider;
        uint32_t                state;
        IOCommandGate         * commandGate;
        IOWorkLoop            * workLoop;
        EventCopyCallerList     eventCopyCallers;
        SetPropertiesCallerList setPropertiesCallers;
        SetLEDCallerList        setLEDCallers;
        OSAction              * eventCopyAction;
        OSAction              * setPropertiesAction;
        OSAction              * setLEDAction;
    };
    
    AppleUserHIDEventService_IVars  *ivar;

protected:

    virtual void    completeCopyEvent(OSAction * action, IOHIDEvent * event, uint64_t context) APPLE_KEXT_OVERRIDE;

public:
    
    virtual IOService *probe(IOService *provider, SInt32 *score) APPLE_KEXT_OVERRIDE;
    virtual bool init(OSDictionary * dictionary = 0) APPLE_KEXT_OVERRIDE;
    virtual void free(void) APPLE_KEXT_OVERRIDE;

    virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;
    virtual void handleStop(IOService * provider) APPLE_KEXT_OVERRIDE;
    virtual bool terminate(IOOptionBits options = 0) APPLE_KEXT_OVERRIDE;
    
    // IOHIDEventService overrides
    virtual IOReturn setElementValue(UInt32 usagePage,
                                     UInt32 usage,
                                     UInt32 value) APPLE_KEXT_OVERRIDE;
    virtual OSArray *getReportElements(void) APPLE_KEXT_OVERRIDE;
    virtual bool handleStart(IOService *provider) APPLE_KEXT_OVERRIDE;
    virtual OSString *getTransport(void) APPLE_KEXT_OVERRIDE;
    virtual UInt32 getLocationID(void) APPLE_KEXT_OVERRIDE;
    virtual UInt32 getVendorID(void) APPLE_KEXT_OVERRIDE;
    virtual UInt32 getVendorIDSource(void) APPLE_KEXT_OVERRIDE;
    virtual UInt32 getProductID(void) APPLE_KEXT_OVERRIDE;
    virtual UInt32 getVersion(void) APPLE_KEXT_OVERRIDE;
    virtual UInt32 getCountryCode(void) APPLE_KEXT_OVERRIDE;
    virtual OSString *getManufacturer(void) APPLE_KEXT_OVERRIDE;
    virtual OSString *getProduct(void) APPLE_KEXT_OVERRIDE;
    virtual OSString *getSerialNumber(void) APPLE_KEXT_OVERRIDE;
    
    
    virtual void     dispatchKeyboardEvent(AbsoluteTime                timeStamp,
                                           UInt32                      usagePage,
                                           UInt32                      usage,
                                           UInt32                      value,
                                           IOOptionBits                options = 0) APPLE_KEXT_OVERRIDE;
    
    virtual void     dispatchScrollWheelEventWithFixed(AbsoluteTime                timeStamp,
                                                       IOFixed                     deltaAxis1,
                                                       IOFixed                     deltaAxis2,
                                                       IOFixed                     deltaAxis3,
                                                       IOOptionBits                options = 0) APPLE_KEXT_OVERRIDE;
    
    virtual void    dispatchEvent(IOHIDEvent * event, IOOptionBits options=0) APPLE_KEXT_OVERRIDE;
    
    virtual IOHIDEvent *    copyEvent(
                                IOHIDEventType              type,
                                IOHIDEvent *                matching = 0,
                                IOOptionBits                options = 0) APPLE_KEXT_OVERRIDE;
    
    virtual IOHIDEvent *copyMatchingEvent(OSDictionary *matching) APPLE_KEXT_OVERRIDE;

    virtual IOReturn setProperties(OSObject * properties) APPLE_KEXT_OVERRIDE;
    virtual void completeSetProperties(OSAction * action, IOReturn status, uint64_t context) APPLE_KEXT_OVERRIDE;
    virtual IOReturn setSystemProperties(OSDictionary * properties) APPLE_KEXT_OVERRIDE;
    virtual void completeSetLED(OSAction * action, IOReturn status, uint64_t context) APPLE_KEXT_OVERRIDE;

private:
    void updateElementsProperty(OSArray * userElements, OSArray * deviceElements);
    void setSensorProperties(OSDictionary * sensorProps, OSArray * deviceElements);
    void setDigitizerProperties(OSDictionary * digitizerProps, OSArray * deviceElements);
    void setUnicodeProperties(OSDictionary * unicodeProps, OSArray * deviceElements);
};
#endif /* !_APPLEUSERHIDEVENTSERVICE_H */
