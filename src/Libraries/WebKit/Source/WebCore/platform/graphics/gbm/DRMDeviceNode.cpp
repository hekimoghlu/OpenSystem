/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include "DRMDeviceNode.h"

#if USE(LIBDRM)
#include <fcntl.h>
#include <unistd.h>
#include <wtf/SafeStrerror.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/WTFString.h>

#if USE(GBM)
#include <gbm.h>
#endif

namespace WebCore {

RefPtr<DRMDeviceNode> DRMDeviceNode::create(CString&& filename)
{
    RELEASE_ASSERT(isMainThread());
    return adoptRef(*new DRMDeviceNode(WTFMove(filename)));
}

DRMDeviceNode::DRMDeviceNode(CString&& filename)
    : m_filename(WTFMove(filename))
{
}

DRMDeviceNode::~DRMDeviceNode()
{
#if USE(GBM)
    if (m_gbmDevice.has_value() && m_gbmDevice.value())
        gbm_device_destroy(m_gbmDevice.value());
#endif
}

#if USE(GBM)
struct gbm_device* DRMDeviceNode::gbmDevice() const
{
    if (m_gbmDevice)
        return m_gbmDevice.value();

    RELEASE_ASSERT(isMainThread());
    m_fd = UnixFileDescriptor { open(m_filename.data(), O_RDWR | O_CLOEXEC), UnixFileDescriptor::Adopt };
    if (m_fd) {
        m_gbmDevice = gbm_create_device(m_fd.value());
        if (m_gbmDevice.value())
            return m_gbmDevice.value();

        WTFLogAlways("Failed to create GBM device for DRM node: %s: %s", m_filename.data(), safeStrerror(errno).data());
        m_fd = { };
    } else {
        WTFLogAlways("Failed to open DRM node %s: %s", m_filename.data(), safeStrerror(errno).data());
        m_gbmDevice = nullptr;
    }

    return m_gbmDevice.value();
}
#endif

} // namespace WebCore

#endif // USE(LIBDRM)
