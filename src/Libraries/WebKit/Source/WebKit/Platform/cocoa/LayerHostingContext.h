/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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

#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS CALayer;
OBJC_CLASS CAContext;

#if USE(EXTENSIONKIT)
OBJC_CLASS BELayerHierarchy;
OBJC_CLASS BELayerHierarchyHandle;
OBJC_CLASS BELayerHierarchyHostingTransactionCoordinator;
#endif

namespace WTF {
class MachSendRight;
}

namespace WebKit {

#if USE(EXTENSIONKIT)
constexpr auto contextIDKey = "cid";
constexpr auto processIDKey = "pid";
constexpr auto machPortKey = "p";
#endif

using LayerHostingContextID = uint32_t;
enum class LayerHostingMode : uint8_t;

struct LayerHostingContextOptions {
#if PLATFORM(IOS_FAMILY)
    bool canShowWhileLocked { false };
#endif
#if USE(EXTENSIONKIT)
    bool useHostable { false };
#endif
};

class LayerHostingContext {
    WTF_MAKE_TZONE_ALLOCATED(LayerHostingContext);
    WTF_MAKE_NONCOPYABLE(LayerHostingContext);
public:
    static std::unique_ptr<LayerHostingContext> createForPort(const WTF::MachSendRight& serverPort);
    
#if HAVE(OUT_OF_PROCESS_LAYER_HOSTING)
    static std::unique_ptr<LayerHostingContext> createForExternalHostingProcess(const LayerHostingContextOptions& = { });

#if PLATFORM(MAC)
    static std::unique_ptr<LayerHostingContext> createForExternalPluginHostingProcess();
#endif
    
    static std::unique_ptr<LayerHostingContext> createTransportLayerForRemoteHosting(LayerHostingContextID);

    static RetainPtr<CALayer> createPlatformLayerForHostingContext(LayerHostingContextID);

#endif // HAVE(OUT_OF_PROCESS_LAYER_HOSTING)

    LayerHostingContext();
    ~LayerHostingContext();

    void setRootLayer(CALayer *);
    CALayer *rootLayer() const;

    LayerHostingContextID contextID() const;
    void invalidate();

    LayerHostingMode layerHostingMode() { return m_layerHostingMode; }

    void setColorSpace(CGColorSpaceRef);
    CGColorSpaceRef colorSpace() const;

#if PLATFORM(MAC)
    void setColorMatchUntaggedContent(bool);
    bool colorMatchUntaggedContent() const;
#endif

    // Fences only work on iOS and OS 10.10+.
    void setFencePort(mach_port_t);

    // createFencePort does not install the fence port on the LayerHostingContext's
    // CAContext; call setFencePort() with the newly created port if synchronization
    // with this context is desired.
    WTF::MachSendRight createFencePort();
    
    // Should be only be used inside webprocess
    void updateCachedContextID(LayerHostingContextID);
    LayerHostingContextID cachedContextID();

#if USE(EXTENSIONKIT)
    OSObjectPtr<xpc_object_t> xpcRepresentation() const;
    RetainPtr<BELayerHierarchy> hostable() const { return m_hostable; }

    static RetainPtr<BELayerHierarchyHandle> createHostingHandle(uint64_t pid, uint64_t contextID);
    static RetainPtr<BELayerHierarchyHostingTransactionCoordinator> createHostingUpdateCoordinator(mach_port_t sendRight);
#endif

private:
    LayerHostingMode m_layerHostingMode;
    // Denotes the contextID obtained from GPU process, should be returned
    // for all calls to context ID in web process when UI side compositing
    // is enabled. This is done to avoid making calls to CARenderServer from webprocess
    LayerHostingContextID m_cachedContextID;
    RetainPtr<CAContext> m_context;
#if USE(EXTENSIONKIT)
    RetainPtr<BELayerHierarchy> m_hostable;
#endif
};

} // namespace WebKit

