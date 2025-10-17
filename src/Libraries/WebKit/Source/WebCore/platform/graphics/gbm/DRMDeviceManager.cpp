/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#include "DRMDeviceManager.h"

#include <wtf/text/WTFString.h>

#if USE(LIBDRM)
#include "DRMDeviceNode.h"
#include <xf86drm.h>

namespace WebCore {

DRMDeviceManager& DRMDeviceManager::singleton()
{
    static std::unique_ptr<DRMDeviceManager> s_manager;
    static std::once_flag s_onceFlag;
    std::call_once(s_onceFlag, [] {
        s_manager = makeUnique<DRMDeviceManager>();
    });
    return *s_manager;
}

DRMDeviceManager::~DRMDeviceManager() = default;

static void drmForeachDevice(Function<bool(drmDevice*)>&& functor)
{
    std::array<drmDevicePtr, 64> devices = { };

    int numDevices = drmGetDevices2(0, devices.data(), std::size(devices));
    if (numDevices <= 0)
        return;

    for (int i = 0; i < numDevices; ++i) {
        if (!functor(devices[i]))
            break;
    }
    drmFreeDevices(devices.data(), numDevices);
}

void DRMDeviceManager::initializeMainDevice(const String& deviceFile)
{
    RELEASE_ASSERT(isMainThread());
    RELEASE_ASSERT(!m_mainDevice.isInitialized);
    m_mainDevice.isInitialized = true;
    if (deviceFile.isEmpty())
        return;

    drmForeachDevice([&](drmDevice* device) {
        const auto nodes = unsafeMakeSpan(device->nodes, DRM_NODE_MAX);
        for (int i = 0; i < DRM_NODE_MAX; ++i) {
            if (!(device->available_nodes & (1 << i)))
                continue;

            if (String::fromUTF8(nodes[i]) == deviceFile) {
                RELEASE_ASSERT(device->available_nodes & (1 << DRM_NODE_PRIMARY));
                if (device->available_nodes & (1 << DRM_NODE_RENDER)) {
                    m_mainDevice.primaryNode = DRMDeviceNode::create(CString { nodes[DRM_NODE_PRIMARY] });
                    m_mainDevice.renderNode = DRMDeviceNode::create(CString { nodes[DRM_NODE_RENDER] });
                } else
                    m_mainDevice.primaryNode = DRMDeviceNode::create(CString { nodes[DRM_NODE_PRIMARY] });
                return false;
            }
        }
        return true;
    });

    if (!m_mainDevice.primaryNode)
        WTFLogAlways("Failed to find DRM device for %s", deviceFile.utf8().data());
}

RefPtr<DRMDeviceNode> DRMDeviceManager::mainDeviceNode(DRMDeviceManager::NodeType nodeType) const
{
    RELEASE_ASSERT(m_mainDevice.isInitialized);

    if (nodeType == NodeType::Render)
        return m_mainDevice.renderNode ? m_mainDevice.renderNode : m_mainDevice.primaryNode;

    return m_mainDevice.primaryNode ? m_mainDevice.primaryNode : m_mainDevice.renderNode;
}

#if USE(GBM)
struct gbm_device* DRMDeviceManager::mainGBMDeviceNode(NodeType nodeType) const
{
    auto node = mainDeviceNode(nodeType);
    return node ? node->gbmDevice() : nullptr;
}
#endif

RefPtr<DRMDeviceNode> DRMDeviceManager::deviceNode(const CString& filename)
{
    RELEASE_ASSERT(isMainThread());
    RELEASE_ASSERT(m_mainDevice.isInitialized);

    if (filename.isNull())
        return nullptr;

    auto node = [&] -> RefPtr<DRMDeviceNode> {
        if (m_mainDevice.primaryNode && m_mainDevice.primaryNode->filename() == filename)
            return m_mainDevice.primaryNode;

        if (m_mainDevice.renderNode && m_mainDevice.renderNode->filename() == filename)
            return m_mainDevice.renderNode;

        return DRMDeviceNode::create(CString { filename.data() });
    }();

#if USE(GBM)
    // Make sure GBMDevice is created in the main thread.
    if (node)
        node->gbmDevice();
#endif

    return node;
}

} // namespace WebCore

#endif // USE(LIBDRM)
