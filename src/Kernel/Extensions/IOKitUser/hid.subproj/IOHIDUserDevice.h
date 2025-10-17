/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#ifndef _IOKIT_HID_IOHIDUSERDEVICE_USER_H
#define _IOKIT_HID_IOHIDUSERDEVICE_USER_H

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <IOKit/hidsystem/IOHIDUserDevice.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

typedef IOReturn (*IOHIDUserDeviceReportCallback)(void * _Nullable refcon, IOHIDReportType type, uint32_t reportID, uint8_t * report, CFIndex reportLength);
typedef IOReturn (*IOHIDUserDeviceReportWithReturnLengthCallback)(void * _Nullable refcon, IOHIDReportType type, uint32_t reportID, uint8_t * report, CFIndex * pReportLength);
typedef IOReturn (*IOHIDUserDeviceHandleReportAsyncCallback)(void * _Nullable refcon, IOReturn result);

static const IOOptionBits kIOHIDUserDeviceCreateOptionStartWhenScheduled = (1<<0);

/*!
    @function   IOHIDUserDeviceCreate
    @abstract   Creates an virtual IOHIDDevice in the kernel.
    @discussion The io_service_t passed in this method must reference an object 
                in the kernel of type IOHIDUserDevice.
    @param      allocator Allocator to be used during creation.
    @param      properties CFDictionaryRef containing device properties index by keys defined in IOHIDKeys.h.
    @result     Returns a new IOHIDUserDeviceRef.
*/
CF_EXPORT
IOHIDUserDeviceRef IOHIDUserDeviceCreate(CFAllocatorRef _Nullable allocator, CFDictionaryRef properties);

/*!
     @function   IOHIDUserDeviceCopyService
     @abstract   Returns the io_service_t for an IOHIDUserDevice, if it has one.
     @discussion If the IOHIDUserDevice references an object in the kernel, this is
                 used to get the io_service_t for that object.
     @param      device Reference to an IOHIDUserDevice.
     @result     Returns the io_service_t if the IOHIDUserDevice has one, or
                 MACH_PORT_NULL if it does not. Object handle should be released with released with IOObjectRelease.
 */
CF_EXPORT
io_service_t IOHIDUserDeviceCopyService(IOHIDUserDeviceRef device);

/*!
 @function   IOHIDUserDeviceCreateWithOptions
 @abstract   Creates an virtual IOHIDDevice in the kernel.
 @discussion The io_service_t passed in this method must reference an object
 in the kernel of type IOHIDUserDevice.  Please use the kIOHIDUserDeviceCreateOptionStartWhenScheduled option 
 if you would like to ensure that callbacks are in place and scheduled prior to starting the device.  This will
 that you will be able to handle getReport and setReport operations if performed during creation.
 @param      allocator Allocator to be used during creation.
 @param      properties CFDictionaryRef containing device properties index by keys defined in IOHIDKeys.h.
 @param      options options to be used when allocating the device
 @result     Returns a new IOHIDUserDeviceRef.
 */
CF_EXPORT
IOHIDUserDeviceRef IOHIDUserDeviceCreateWithOptions(CFAllocatorRef _Nullable allocator, CFDictionaryRef properties, IOOptionBits options);


/*!
    @function   IOHIDUserDeviceScheduleWithRunLoop
    @abstract   Schedules the IOHIDUserDevice with a run loop
    @discussion This is necessary to receive asynchronous events from the kernel
    @param      device Reference to IOHIDUserDevice 
    @param      runLoop Run loop to be scheduled with
    @param      runLoopMode Run loop mode to be scheduled with
*/
CF_EXPORT
void IOHIDUserDeviceScheduleWithRunLoop(IOHIDUserDeviceRef device, CFRunLoopRef runLoop, CFStringRef runLoopMode);

/*!
    @function   IOHIDUserDeviceUnscheduleFromRunLoop
    @abstract   Unschedules the IOHIDUserDevice from a run loop
    @param      device Reference to IOHIDUserDevice 
    @param      runLoop Run loop to be scheduled with
    @param      runLoopMode Run loop mode to be scheduled with
*/
CF_EXPORT
void IOHIDUserDeviceUnscheduleFromRunLoop(IOHIDUserDeviceRef device, CFRunLoopRef runLoop, CFStringRef runLoopMode);

/*!
 * @function IOHIDUserDeviceScheduleWithDispatchQueue
 *
 * @abstract
 * Schedules the IOHIDUserDevice with a dispatch queue in order to receive
 * asynchronous events from the kernel.
 *
 * @discussion
 * After calling IOHIDUserDeviceScheduleWithDispatchQueue, the queue is
 * considered active. All "Register" functions should be called before
 * scheduling with a dispatch queue.
 *
 * An IOHIDUserDevice should not be scheduled with both a run loop
 * and dispatch queue. IOHIDUserDeviceScheduleWithDispatchQueue should
 * not be called more than once.
 *
 * A call to IOHIDUserDeviceScheduleWithDispatchQueue should be balanced
 * with a call to IOHIDUserDeviceUnscheduleFromDispatchQueue.
 *
 * @param device
 * Reference to an IOHIDUserDevice.
 *
 * @param queue
 * The dispatch queue to which the event handler block will be submitted.
 */
CF_EXPORT
void IOHIDUserDeviceScheduleWithDispatchQueue(IOHIDUserDeviceRef device, dispatch_queue_t queue);

/*!
 * @function IOHIDUserDeviceUnscheduleFromDispatchQueue
 *
 * @abstract
 * Unschedules the dispatch queue, preventing any further invocation
 * of its event handler block.
 *
 * @discussion
 * Unscheduling prevents any further invocation of the event handler block for
 * the specified dispatch queue, but does not interrupt an event handler
 * block that is already in progress.
 *
 * Explicit unschesduling of the dispatch queue is required, no implicit
 * unscheduling takes place.
 *
 * Calling IOHIDUserDeviceUnscheduleFromDispatchQueue on an already unscheduled
 * device has no effect.
 *
 * @param device
 * Reference to an IOHIDUserDevice.
 *
 * @param queue
 * The dispatch queue to unschedule from.
 */
CF_EXPORT
void IOHIDUserDeviceUnscheduleFromDispatchQueue(IOHIDUserDeviceRef device, dispatch_queue_t queue);

/*!
    @function   IOHIDUserDeviceRegisterGetReportCallback
    @abstract   Register a callback to receive get report requests
    @discussion The call to IOHIDUserDeviceRegisterGetReportCallback
                should be made before the device is scheduled with
                a dispatch queue.
    @param      device Reference to IOHIDUserDevice 
    @param      callback Callback of type IOHIDUserDeviceReportCallback to be used
    @param      refcon pointer to a reference object of your choosing
*/
CF_EXPORT
void IOHIDUserDeviceRegisterGetReportCallback(IOHIDUserDeviceRef device, IOHIDUserDeviceReportCallback _Nullable callback, void * _Nullable refcon);

/*!
 @function   IOHIDUserDeviceRegisterGetReportWithLegthCallback
 @abstract   Register a callback to receive get report requests
 @discussion Unlike the callback specified in IOHIDUserDeviceRegisterGetReportCallback,
             the callback passed here allows the callee to return the actual bytes read.
             The call to IOHIDUserDeviceRegisterGetReportWithReturnLengthCallback should
             be made before the device is scheduled with a dispatch queue.
 @param      device Reference to IOHIDUserDevice
 @param      callback Callback of type IOHIDUserDeviceReportWithReturnLengthCallback to be used
 @param      refcon pointer to a reference object of your choosing
 */
CF_EXPORT
void IOHIDUserDeviceRegisterGetReportWithReturnLengthCallback(IOHIDUserDeviceRef device, IOHIDUserDeviceReportWithReturnLengthCallback _Nullable callback, void * _Nullable refcon);

/*!
    @function   IOHIDUserDeviceRegisterSetReportCallback
    @abstract   Register a callback to receive set report requests
    @discussion The call to IOHIDUserDeviceRegisterSetReportCallback
                should be made before the device is scheduled with
                a dispatch queue.
    @param      device Reference to IOHIDUserDevice 
    @param      callback Callback to be used
    @param      refcon pointer to a reference object of your choosing
*/
CF_EXPORT
void IOHIDUserDeviceRegisterSetReportCallback(IOHIDUserDeviceRef device, IOHIDUserDeviceReportCallback _Nullable callback, void * _Nullable refcon);

/*!
    @function   IOHIDUserDeviceHandleReport
    @abstract   Dispatch a report to the IOHIDUserDevice.
    @param      device Reference to IOHIDUserDevice 
    @param      report Buffer containing formated report being issued to HID stack
    @param      reportLength Report buffer length
    @result     Returns kIOReturnSuccess when report is handled successfully.
*/
CF_EXPORT
IOReturn IOHIDUserDeviceHandleReport(IOHIDUserDeviceRef device, const uint8_t * report, CFIndex reportLength);

/*!
    @function   IOHIDUserDeviceHandleReportAsync
    @abstract   Dispatch a report to the IOHIDUserDevice.
    @param      device Reference to IOHIDUserDevice 
    @param      report Buffer containing formated report being issued to HID stack
    @param      reportLength Report buffer length
    @param      callback Callback to be used (optional)
    @param      refcon pointer to a reference object of your choosing (optional)
    @result     Returns kIOReturnSuccess when report is handled successfully.
 */
CF_EXPORT
IOReturn IOHIDUserDeviceHandleReportAsync(IOHIDUserDeviceRef device, const uint8_t *report, CFIndex reportLength, IOHIDUserDeviceHandleReportAsyncCallback _Nullable callback, void * _Nullable refcon);

/*!
    @function   IOHIDUserDeviceHandleReportAsync
    @abstract   Dispatch a report to the IOHIDUserDevice.
    @param      device Reference to IOHIDUserDevice
    @param      timestamp mach_absolute_time() based timestamp
    @param      report Buffer containing formated report being issued to HID stack
    @param      reportLength Report buffer length
    @param      callback Callback to be used (optional)
    @param      refcon pointer to a reference object of your choosing (optional)
    @result     Returns kIOReturnSuccess when report is handled successfully.
 */
CF_EXPORT
IOReturn IOHIDUserDeviceHandleReportAsyncWithTimeStamp(IOHIDUserDeviceRef device, uint64_t timestamp, const uint8_t *report, CFIndex reportLength, IOHIDUserDeviceHandleReportAsyncCallback _Nullable callback, void * _Nullable refcon);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* _IOKIT_HID_IOHIDUSERDEVICE_USER_H */
