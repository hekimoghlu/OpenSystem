/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#ifndef IOHIDServiceClient_h
#define IOHIDServiceClient_h

#include <CoreFoundation/CoreFoundation.h>

__BEGIN_DECLS
CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

/*!
    @header IOHIDServiceClient
    
    IOHIDServiceClient serves as a client to the HID event system
    services. Users are able to copy/set specific properties (defined in
    <code>IOKit/hid/IOHIDProperties.h</code>), and gather more information
    about the services available in the HID event system.
 */

typedef struct CF_BRIDGED_TYPE(id) __IOHIDServiceClient * IOHIDServiceClientRef;

/*!
 * @function    IOHIDServiceClientSetProperty
 *
 * @abstract    Sets a property on the HID service.
 *
 * @param       service the HID service to set the property on.
 *
 * @param       key the property key to set. A list of keys can be found in <code>HIDProperties.h</code>.
 *
 * @param       property the value to set the property.
 *
 * @result      Returns true on success.
 */
Boolean IOHIDServiceClientSetProperty(IOHIDServiceClientRef service, CFStringRef key, CFTypeRef property);

/*!
 * @function    IOHIDServiceClientCopyProperty
 *
 * @abstract    Copies a property from the HID service.
 *
 * @param       service the HID service to copy the property from.
 *
 * @param       key the property key to copy. A list of keys can be found in <code>HIDProperties.h</code>.
 *
 * @result      Returns a CFTypeRef of the property to be copied on success, otherwise NULL.
 *              Caller is responsible for calling CFRelease on the property.
 */
CFTypeRef _Nullable IOHIDServiceClientCopyProperty(IOHIDServiceClientRef service, CFStringRef key);

/*!
 * @function    IOHIDServiceClientGetTypeID
 *
 * @result      Returns the CFTypeID of the <code>IOHIDServiceClient</code> class.
 */
CFTypeID IOHIDServiceClientGetTypeID(void);

/*!
 * @function    IOHIDServiceClientGetRegistryID
 *
 * @param       service the HID service to get the registry ID for.
 *
 * @result      Returns a CFTypeRef containing the registry ID for the service.
 */
CFTypeRef IOHIDServiceClientGetRegistryID(IOHIDServiceClientRef service);

/*!
 * @function    IOHIDServiceClientConformsTo
 *
 * @abstract    Determines if a HID service conforms to a specific usage page and usage.
 *
 * @param       usagePage A usage page defined in <code>IOHIDUsageTables.h</code>.
 *
 * @param       usage A usage defined in <code>IOHIDUsageTables.h</code>.
 *
 * @result      Returns true if the service conforms to the provided usage page and usage.
 */
boolean_t IOHIDServiceClientConformsTo(IOHIDServiceClientRef service, uint32_t usagePage, uint32_t usage);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END
__END_DECLS

#endif /* IOHIDServiceClient_h */
