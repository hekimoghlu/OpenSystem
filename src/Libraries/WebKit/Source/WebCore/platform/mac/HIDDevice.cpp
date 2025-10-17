/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#include "config.h"
#include "HIDDevice.h"

#if PLATFORM(MAC)

#include "HIDElement.h"
#include "Logging.h"
#include <IOKit/hid/IOHIDElement.h>
#include <wtf/Deque.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/cf/TypeCastsCF.h>

WTF_DECLARE_CF_TYPE_TRAIT(IOHIDElement);

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HIDDevice);

static int getDevicePropertyAsInt(IOHIDDeviceRef device, CFStringRef key)
{
    CFNumberRef cfPropertyValue = checked_cf_cast<CFNumberRef>(IOHIDDeviceGetProperty(device, key));
    int propertyValue = -1;
    if (cfPropertyValue)
        CFNumberGetValue(cfPropertyValue, kCFNumberIntType, &propertyValue);
    return propertyValue;
}

HIDDevice::HIDDevice(IOHIDDeviceRef device)
    : m_rawDevice(device)
{
    int vendorID = getDevicePropertyAsInt(device, CFSTR(kIOHIDVendorIDKey));
    int productID = getDevicePropertyAsInt(device, CFSTR(kIOHIDProductIDKey));

    if (vendorID < 0 || vendorID > std::numeric_limits<uint16_t>::max()) {
        LOG(HID, "Device attached with malformed vendor ID 0x%x. Resetting to 0.", vendorID);
        vendorID = 0;
    }
    if (productID < 0 || productID > std::numeric_limits<uint16_t>::max()) {
        LOG(HID, "Device attached with malformed product ID 0x%x. Resetting to 0.", productID);
        productID = 0;
    }

    m_vendorID = (uint16_t)vendorID;
    m_productID = (uint16_t)productID;

    CFStringRef cfProductName = checked_cf_cast<CFStringRef>(IOHIDDeviceGetProperty(device, CFSTR(kIOHIDProductKey)));
    m_productName = cfProductName ? String(cfProductName) : String("Unknown"_s);
}

Vector<HIDElement> HIDDevice::uniqueInputElementsInDeviceTreeOrder() const
{
    UncheckedKeyHashSet<IOHIDElementCookie> encounteredCookies;
    Deque<IOHIDElementRef> elementQueue;

    RetainPtr<CFArrayRef> elements = adoptCF(IOHIDDeviceCopyMatchingElements(m_rawDevice.get(), NULL, kIOHIDOptionsTypeNone));
    CFIndex count = elements ? CFArrayGetCount(elements.get()) : 0;
    for (CFIndex i = 0; i < count; ++i)
        elementQueue.append(checked_cf_cast<IOHIDElementRef>(CFArrayGetValueAtIndex(elements.get(), i)));

    Vector<HIDElement> result;

    while (!elementQueue.isEmpty()) {
        auto element = elementQueue.takeFirst();
        IOHIDElementCookie cookie = IOHIDElementGetCookie(element);
        if (encounteredCookies.contains(cookie))
            continue;

        switch (IOHIDElementGetType(element)) {
        case kIOHIDElementTypeCollection: {
            auto children = IOHIDElementGetChildren(element);
            for (CFIndex i = CFArrayGetCount(children) - 1; i >= 0; --i)
                elementQueue.prepend(checked_cf_cast<IOHIDElementRef>(CFArrayGetValueAtIndex(children, i)));
            continue;
        }
        case kIOHIDElementTypeInput_Misc:
        case kIOHIDElementTypeInput_Button:
        case kIOHIDElementTypeInput_Axis:
            encounteredCookies.add(cookie);
            result.append(element);
            continue;
        default:
            continue;
        }
    }

    return result;
}

} // namespace WebCore

#endif // PLATFORM(MAC)
