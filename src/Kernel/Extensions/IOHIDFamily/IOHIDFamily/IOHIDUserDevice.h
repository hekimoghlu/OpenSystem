/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
#ifndef _IOHIDUSERDEVICE_H
#define _IOHIDUSERDEVICE_H

/*
 * Kernel
 */

#include <IOKit/hid/IOHIDDevice.h>

class IOHIDResourceDeviceUserClient;

/*! @class IOHIDUserDevice : public IOHIDDevice
    @abstract 
*/
class IOHIDUserDevice : public IOHIDDevice
{
    OSDeclareDefaultStructors(IOHIDUserDevice);

private:
    OSDictionary *                  _properties;
    IOHIDResourceDeviceUserClient * _provider;
    
protected:
/*! @function free
    @abstract Free the IOHIDDevice object.
    @discussion Release all resources that were previously allocated,
    then call super::free() to propagate the call to our superclass. */
    virtual void free(void) APPLE_KEXT_OVERRIDE;

/*! @function handleStart
    @abstract Prepare the hardware and driver to support I/O operations.
    @discussion IOHIDDevice will call this method from start() before
    any I/O operations are issued to the concrete subclass. Methods
    such as newReportDescriptor() are only called after handleStart()
    has returned true. A subclass that overrides this method should
    begin its implementation by calling the version in super, and
    then check the return value.
    @param provider The provider argument passed to start().
    @result True on success, or false otherwise. Returning false will
    cause start() to fail and return false. */
    virtual bool handleStart(IOService * provider) APPLE_KEXT_OVERRIDE;

/*! @function handleStop
    @abstract Quiesce the hardware and stop the driver.
    @discussion IOHIDDevice will call this method from stop() to
    signal that the hardware should be quiesced and the driver stopped.
    A subclass that overrides this method should end its implementation
    by calling the version in super.
    @param provider The provider argument passed to stop(). */
    virtual void handleStop(IOService * provider) APPLE_KEXT_OVERRIDE;

public:
/*! @function withProperties
    @abstract Creates a device with properties.
    @param properties Dictionary containing preflighted properties corresponding to device keys from IOHIDKeys.h.
	@result The new device created.
*/
	static IOHIDUserDevice * withProperties(OSDictionary * properties);
    
/*! @function initWithProperties
    @abstract Initialize an IOHIDUserDevice object.
    @discussion Prime the IOHIDUserDevice object and prepare it to support
    a probe() or a start() call.
    @param properties Dictionary containing preflighted properties corresponding to device keys from IOHIDKeys.h.
    @result True on success, or false otherwise. */
    virtual bool initWithProperties(OSDictionary * properties);

/*! @function newReportDescriptor
    @abstract Create and return a new memory descriptor that describes the
    report descriptor for the HID device.
    @result kIOReturnSuccess on success, or an error return otherwise. */
	virtual IOReturn newReportDescriptor(IOMemoryDescriptor ** descriptor ) const APPLE_KEXT_OVERRIDE;

/*! @function newTransportString
    @abstract Returns a string object that describes the transport
    layer used by the HID device.
    @result A string object. The caller must decrement the retain count
    on the object returned. */
    virtual OSString * newTransportString() const APPLE_KEXT_OVERRIDE;

/*! @function newManufacturerString
    @abstract Returns a string object that describes the manufacturer
    of the HID device.
    @result A string object. The caller must decrement the retain count
    on the object returned. */
    virtual OSString * newManufacturerString() const APPLE_KEXT_OVERRIDE;

/*! @function newProductString
    @abstract Returns a string object that describes the product
    of the HID device.
    @result A string object. The caller must decrement the retain count
    on the object returned. */
    virtual OSString * newProductString() const APPLE_KEXT_OVERRIDE;

/*! @function newVendorIDNumber
    @abstract Returns a number object that describes the vendor ID
    of the HID device.
    @result A number object. The caller must decrement the retain count
    on the object returned. */
    virtual OSNumber * newVendorIDNumber() const APPLE_KEXT_OVERRIDE;

/*! @function newProductIDNumber
    @abstract Returns a number object that describes the product ID
    of the HID device.
    @result A number object. The caller must decrement the retain count
    on the object returned. */
    virtual OSNumber * newProductIDNumber() const APPLE_KEXT_OVERRIDE;

/*! @function newVersionNumber
    @abstract Returns a number object that describes the version number
    of the HID device.
    @result A number object. The caller must decrement the retain count
    on the object returned. */
    virtual OSNumber * newVersionNumber() const APPLE_KEXT_OVERRIDE;

/*! @function newSerialNumberString
    @abstract Returns a string object that describes the serial number
    of the HID device.
    @result A number object. The caller must decrement the retain count
    on the object returned. */
    virtual OSString * newSerialNumberString(void) const APPLE_KEXT_OVERRIDE;

/*! @function newVendorIDSourceNumber
    @abstract Returns a number object that describes the vendor ID
    source of the HID device.  
    @result A number object. The caller must decrement the retain count
    on the object returned. */
    virtual OSNumber * newVendorIDSourceNumber(void) const APPLE_KEXT_OVERRIDE;

/*! @function newCountryCodeNumber
    @abstract Returns a number object that describes the country code
    of the HID device.  
    @result A number object. The caller must decrement the retain count
    on the object returned. */
    virtual OSNumber * newCountryCodeNumber(void) const APPLE_KEXT_OVERRIDE;
    
/*! @function newReportIntervalNumber
    @abstract Returns a number object that describes the report interval
    of the HID device.  
    @result A number object. The caller must decrement the retain count
    on the object returned. */
    virtual OSNumber * newReportIntervalNumber(void) const APPLE_KEXT_OVERRIDE;

    /*! @function newLocationIDNumber
     @abstract Returns a number object that describes the location
     of the HID device.  
     @result A number object. The caller must decrement the retain count
     on the object returned. */
    virtual OSNumber *newLocationIDNumber(void) const APPLE_KEXT_OVERRIDE;

/*! @function getReport
    @abstract Get a report synchronously from the HID device.
    @param report A memory descriptor that describes the memory to store
    the report read from the HID device.
    @param reportType The report type.
    @param options The lower 8 bits will represent the Report ID.  The
    other 24 bits are options to specify the request.
    @result kIOReturnSuccess on success, or an error return otherwise. */
    IOReturn getReport(IOMemoryDescriptor * report,
                       IOHIDReportType      reportType,
                       IOOptionBits         options) APPLE_KEXT_OVERRIDE;

/*! @function getReport
    @abstract Get a report asynchronously from the HID device.
    @param report A memory descriptor that describes the memory to store
    the report read from the HID device.
    @param reportType The report type.
    @param options The lower 8 bits will represent the Report ID.  The
    other 24 bits are options to specify the request.
    @param completionTimeout Maximum amount of time to wait for a
    response before the completion is triggered with an error.
    @param completion Callback info for completing the asynchronous call.
    Will execute synchronously if the completion or callback is null.
    @result kIOReturnSuccess on success, or an error return otherwise. */
    IOReturn getReport(IOMemoryDescriptor * report,
                       IOHIDReportType      reportType,
                       IOOptionBits         options,
                       UInt32               completionTimeout,
                       IOHIDCompletion    * completion = 0) APPLE_KEXT_OVERRIDE;

/*! @function setReport
    @abstract Send a report synchronously to the HID device.
    @param report A memory descriptor that describes the report to send
    to the HID device.
    @param reportType The report type.
    @param options The lower 8 bits will represent the Report ID.  The
    other 24 bits are options to specify the request.
    @result kIOReturnSuccess on success, or an error return otherwise. */
    IOReturn setReport(IOMemoryDescriptor * report,
                       IOHIDReportType      reportType,
                       IOOptionBits         options) APPLE_KEXT_OVERRIDE;

/*! @function setReport
    @abstract Send a report asynchronously to the HID device.
    @param report A memory descriptor that describes the report to send
    to the HID device.
    @param reportType The report type.
    @param options The lower 8 bits will represent the Report ID.  The
    other 24 bits are options to specify the request.
    @param completionTimeout Maximum amount of time to wait for a
    response before the completion is triggered with an error.
    @param completion Callback info for completing the asynchronous call.
    Will execute synchronously if the completion or callback is null.
    @result kIOReturnSuccess on success, or an error return otherwise. */
    IOReturn setReport(IOMemoryDescriptor * report,
                       IOHIDReportType      reportType,
                       IOOptionBits         options,
                       UInt32               completionTimeout,
                       IOHIDCompletion    * completion = 0) APPLE_KEXT_OVERRIDE;

};


#endif /* !_IOHIDUSERDEVICE_H */

