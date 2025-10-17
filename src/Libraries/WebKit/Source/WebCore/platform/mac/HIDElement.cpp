/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#include "HIDElement.h"

#if PLATFORM(MAC)

#include "Logging.h"
#include <IOKit/hid/IOHIDElement.h>
#include <IOKit/hid/IOHIDValue.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HIDElement);

HIDElement::HIDElement(IOHIDElementRef element)
    : m_physicalMin(IOHIDElementGetPhysicalMin(element))
    , m_physicalMax(IOHIDElementGetPhysicalMax(element))
    , m_physicalValue(m_physicalMin)
    , m_usage(IOHIDElementGetUsage(element))
    , m_usagePage(IOHIDElementGetUsagePage(element))
    , m_cookie(IOHIDElementGetCookie(element))
    , m_rawElement(element)
{
}

void HIDElement::valueChanged(IOHIDValueRef value)
{
    if (IOHIDValueGetElement(value) != m_rawElement.get()) {
        LOG(HID, "HIDElement: Changed value whose IOHIDElement %p doesn't match %p", IOHIDValueGetElement(value), m_rawElement.get());
        return;
    }

    m_physicalValue = IOHIDValueGetScaledValue(value, kIOHIDValueScaleTypePhysical);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
