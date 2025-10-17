/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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

#if HAVE(SEC_ACCESS_CONTROL)

#import <wtf/RetainPtr.h>
#import <wtf/cf/VectorCF.h>
#import <wtf/spi/cocoa/SecuritySPI.h>

namespace WebKit {

class CoreIPCSecAccessControl {
public:
    CoreIPCSecAccessControl(SecAccessControlRef accessControl)
        : m_accessControlData(dataFromAccessControl(accessControl))
    {
    }

    CoreIPCSecAccessControl(RetainPtr<CFDataRef> data)
        : m_accessControlData(data)
    {
    }

    CoreIPCSecAccessControl(std::span<const uint8_t> data)
        : m_accessControlData(adoptCF(CFDataCreate(kCFAllocatorDefault, data.data(), data.size())))
    {
    }

    RetainPtr<SecAccessControlRef> createSecAccessControl() const
    {
        auto accessControl = adoptCF(SecAccessControlCreateFromData(kCFAllocatorDefault, m_accessControlData.get(), NULL));
        ASSERT(accessControl);
        return accessControl;
    }

    std::span<const uint8_t> dataReference() const
    {
        if (!m_accessControlData)
            return { };
        return span(m_accessControlData.get());
    }

private:
    RetainPtr<CFDataRef> dataFromAccessControl(SecAccessControlRef accessControl) const
    {
        ASSERT(accessControl);
        auto data = adoptCF(SecAccessControlCopyData(accessControl));
        ASSERT(data);
        return data;
    }

    RetainPtr<CFDataRef> m_accessControlData;
};

} // namespace WebKit

#endif // HAVE(SEC_ACCESS_CONTROL)
