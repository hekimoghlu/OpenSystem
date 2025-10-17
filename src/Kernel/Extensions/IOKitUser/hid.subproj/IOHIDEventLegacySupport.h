/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#ifndef _IOKIT_HID_IOHIDEVENTLEGACYSUPPORT_H
#define _IOKIT_HID_IOHIDEVENTLEGACYSUPPORT_H

#include <IOKit/hid/IOHIDEventData.h>
#include <IOKit/hid/IOHIDEventTypes.h>

/* NOTE: The following methods are applicable only on VisionOS; on all other platforms they behave as no-ops*/

/*!
    @function   IOHIDEventHasLegacyEventData
    @discussion Check if an event has associated legacy data to be used during serialization
    @param      type The type of the HID event.
    @result     True if the event has legacy data, false if not.
*/
bool            __IOHIDEventHasLegacyEventData(IOHIDEventType type);

/*!
    @function   IOHIDEventDataAppendFromLegacyEvent
    @discussion Called if an event has associated legacy data (and legacy is enabled); uses legacy data representation to translate from non-legacy data and store event data in the buffer
    @param      eventData Non-legacy event data to be translated
    @param      buffer  Data buffer to be populated with legacy data
    @result     Size of the data appended to the buffer.
*/
CFIndex         __IOHIDEventDataAppendFromLegacyEvent(IOHIDEventData * eventData, UInt8* buffer);

/*!
    @function   IOHIDEventPopulateCurrentEventData
    @discussion Called if an event has associated legacy data (and legacy is enabled); uses non-legacy data representation to translate from legacy data and store event data in the buffer
    @param      eventData Legacy event data to be translated
    @param      buffer  Data buffer to be populated with non-legacy data
*/
void            __IOHIDEventPopulateCurrentEventData(IOHIDEventData * eventData, IOHIDEventData * newEventData);


#endif /* _IOKIT_HID_IOHIDEVENTLEGACYSUPPORT_H */
