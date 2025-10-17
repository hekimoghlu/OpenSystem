/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#pragma once

#if PLATFORM(MAC)

#include <IOKit/hid/IOHIDDevice.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class HIDElement {
    WTF_MAKE_TZONE_ALLOCATED(HIDElement);
public:
    explicit HIDElement(IOHIDElementRef);

    IOHIDElementRef rawElement() const { return m_rawElement.get(); }

    CFIndex physicalMin() const { return m_physicalMin; }
    CFIndex physicalMax() const { return m_physicalMax; }
    CFIndex physicalValue() const { return m_physicalValue; }
    uint32_t usage() const { return m_usage; }
    uint32_t usagePage() const { return m_usagePage; }
    uint64_t fullUsage() const { return ((uint64_t)m_usagePage) << 32 | m_usage; }
    IOHIDElementCookie cookie() const { return m_cookie; }

    void valueChanged(IOHIDValueRef);

private:
    CFIndex m_physicalMin;
    CFIndex m_physicalMax;
    CFIndex m_physicalValue;
    uint32_t m_usage;
    uint32_t m_usagePage;
    IOHIDElementCookie m_cookie;
    RetainPtr<IOHIDElementRef> m_rawElement;
};


} // namespace WebCore

#endif // PLATFORM(MAC)
