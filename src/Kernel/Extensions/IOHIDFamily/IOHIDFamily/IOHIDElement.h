/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#ifndef _IOKIT_HID_IOHIDELEMENT_H
#define _IOKIT_HID_IOHIDELEMENT_H

#include <libkern/c++/OSArray.h>
#include <libkern/c++/OSData.h>
#include <IOKit/hid/IOHIDKeys.h>

//===========================================================================
// An object that describes a single HID element.

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHIDElement : public OSCollection
#else
class IOHIDElement : public OSCollection
#endif
{
    OSDeclareAbstractStructors( IOHIDElement )
    
    /* Internal use only  */
    struct ExpansionData { 
    };
    ExpansionData *             _reserved;


public:

    virtual IOHIDElementCookie              getCookie()            = 0;
    virtual IOHIDElement *                  getParentElement()     = 0;
    virtual OSArray *                       getChildElements()     = 0;
    virtual IOHIDElementType                getType()              = 0;
    virtual IOHIDElementCollectionType      getCollectionType()    = 0;
    virtual UInt32                          getUsagePage()         = 0;
    virtual UInt32                          getUsage()             = 0;
    virtual UInt32                          getLogicalMin()        = 0;
    virtual UInt32                          getLogicalMax()        = 0;
    virtual UInt32                          getPhysicalMin()       = 0;
    virtual UInt32                          getPhysicalMax()       = 0;
    virtual UInt32                          getUnitExponent()      = 0;
    virtual UInt32                          getUnit()              = 0;
    virtual UInt32                          getReportSize()        = 0;
    virtual UInt32                          getReportCount()       = 0;
    virtual UInt32                          getReportID()          = 0;
    virtual UInt32                          getFlags()             = 0;
    virtual AbsoluteTime                    getTimeStamp()         = 0;
    virtual UInt32                          getValue()             = 0;
    virtual OSData *                        getDataValue()         = 0;
    virtual void                            setValue(UInt32 value)  = 0;
    virtual void                            setDataValue(OSData * value) = 0;
  
    OSMetaClassDeclareReservedUsed(IOHIDElement,  0);
    virtual bool                            conformsTo(UInt32 usagePage, UInt32 usage=0) = 0;
    
    OSMetaClassDeclareReservedUsed(IOHIDElement,  1);
    virtual void                            setCalibration(UInt32 min=0, UInt32 max=0, UInt32 saturationMin=0, UInt32 saturationMax=0, UInt32 deadZoneMin=0, UInt32 deadZoneMax=0, IOFixed granularity=0) = 0;
    
    OSMetaClassDeclareReservedUsed(IOHIDElement,  2);
    virtual UInt32                          getScaledValue(IOHIDValueScaleType type=kIOHIDValueScaleTypePhysical) = 0;
    
    OSMetaClassDeclareReservedUsed(IOHIDElement,  3);
    virtual IOFixed                         getScaledFixedValue(IOHIDValueScaleType type=kIOHIDValueScaleTypePhysical) = 0;

    OSMetaClassDeclareReservedUsed(IOHIDElement,  4);
    virtual UInt32                          getValue(IOOptionBits options) = 0;

    OSMetaClassDeclareReservedUsed(IOHIDElement,  5);
    virtual OSData *                        getDataValue(IOOptionBits options) = 0;
    
    OSMetaClassDeclareReservedUsed(IOHIDElement,  6);
    virtual boolean_t                       isVariableSize()        = 0;

    OSMetaClassDeclareReservedUsed(IOHIDElement,  7);
    virtual IOFixed                         getScaledFixedValue(IOHIDValueScaleType type, IOOptionBits options)  = 0;

    OSMetaClassDeclareReservedUsed(IOHIDElement,  8);
    virtual void                            setValue(UInt32 value, IOOptionBits options) = 0;
    OSMetaClassDeclareReservedUsed(IOHIDElement,  9);
    virtual bool                            getReportType(IOHIDReportType * reportType) const = 0;
    OSMetaClassDeclareReservedUnused(IOHIDElement, 10);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 11);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 12);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 13);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 14);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 15);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 16);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 17);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 18);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 19);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 20);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 21);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 22);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 23);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 24);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 25);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 26);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 27);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 28);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 29);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 30);
    OSMetaClassDeclareReservedUnused(IOHIDElement, 31);

};

#endif /* !_IOKIT_HID_IOHIDELEMENT_H */
