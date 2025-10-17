/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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

#if USE(LIBDRM)

#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/CString.h>
#include <wtf/unix/UnixFileDescriptor.h>

#if USE(GBM)
struct gbm_device;
#endif

namespace WTF {
class String;
}

namespace WebCore {

class DRMDeviceNode : public ThreadSafeRefCounted<DRMDeviceNode, WTF::DestructionThread::Main> {
public:
    static RefPtr<DRMDeviceNode> create(CString&&);
    ~DRMDeviceNode();

    const CString& filename() const { return m_filename; }

#if USE(GBM)
    struct gbm_device* gbmDevice() const;
#endif

private:
    explicit DRMDeviceNode(CString&&);

    CString m_filename;
#if USE(GBM)
    mutable WTF::UnixFileDescriptor m_fd;
    mutable std::optional<struct gbm_device*> m_gbmDevice;
#endif
};

} // namespace WebCore

#endif // USE(LIBDRM)
