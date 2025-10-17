/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>

#if USE(GBM)
struct gbm_device;
#endif

namespace WTF {
class CString;
class String;
}

namespace WebCore {

class DRMDeviceNode;

class DRMDeviceManager {
    WTF_MAKE_NONCOPYABLE(DRMDeviceManager);
    WTF_MAKE_FAST_ALLOCATED();
public:
    static DRMDeviceManager& singleton();

    DRMDeviceManager() = default;
    ~DRMDeviceManager();

    void initializeMainDevice(const WTF::String&);
    bool isInitialized() const { return m_mainDevice.isInitialized; }

    enum class NodeType : bool { Primary, Render };
    RefPtr<DRMDeviceNode> mainDeviceNode(NodeType) const;
    RefPtr<DRMDeviceNode> deviceNode(const WTF::CString&);

#if USE(GBM)
    struct gbm_device* mainGBMDeviceNode(NodeType) const;
#endif

private:
    struct {
        bool isInitialized { false };
        RefPtr<DRMDeviceNode> primaryNode;
        RefPtr<DRMDeviceNode> renderNode;
    } m_mainDevice;
};

} // namespace WebCore

#endif // USE(LIBDRM)
