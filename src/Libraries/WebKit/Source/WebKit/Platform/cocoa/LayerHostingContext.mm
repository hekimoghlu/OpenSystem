/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#import "config.h"
#import "LayerHostingContext.h"

#import "LayerTreeContext.h"
#import <WebCore/WebCoreCALayerExtras.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/MachSendRight.h>
#import <wtf/TZoneMallocInlines.h>

#if USE(EXTENSIONKIT)
#import "ExtensionKitSPI.h"
#import <BrowserEngineKit/BELayerHierarchy.h>
#import <BrowserEngineKit/BELayerHierarchyHandle.h>
#import <BrowserEngineKit/BELayerHierarchyHostingTransactionCoordinator.h>

SOFT_LINK_FRAMEWORK_OPTIONAL(BrowserEngineKit);
SOFT_LINK_CLASS_OPTIONAL(BrowserEngineKit, BELayerHierarchy);
SOFT_LINK_CLASS_OPTIONAL(BrowserEngineKit, BELayerHierarchyHandle);
SOFT_LINK_CLASS_OPTIONAL(BrowserEngineKit, BELayerHierarchyHostingTransactionCoordinator);
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LayerHostingContext);

std::unique_ptr<LayerHostingContext> LayerHostingContext::createForPort(const MachSendRight& serverPort)
{
    auto layerHostingContext = makeUnique<LayerHostingContext>();

    NSDictionary *options = @{
        kCAContextPortNumber : @(serverPort.sendRight()),
#if PLATFORM(MAC)
        kCAContextCIFilterBehavior : @"ignore",
#endif
    };

    layerHostingContext->m_layerHostingMode = LayerHostingMode::InProcess;
    layerHostingContext->m_context = [CAContext remoteContextWithOptions:options];
    layerHostingContext->m_cachedContextID = layerHostingContext->contextID();
    return layerHostingContext;
}

#if HAVE(OUT_OF_PROCESS_LAYER_HOSTING)
std::unique_ptr<LayerHostingContext> LayerHostingContext::createForExternalHostingProcess(const LayerHostingContextOptions& options)
{
    auto layerHostingContext = makeUnique<LayerHostingContext>();
    layerHostingContext->m_layerHostingMode = LayerHostingMode::OutOfProcess;

#if PLATFORM(IOS_FAMILY) && !PLATFORM(MACCATALYST)
    // Use a very large display ID to ensure that the context is never put on-screen
    // without being explicitly parented. See <rdar://problem/16089267> for details.
    auto contextOptions = @{
        kCAContextSecure: @(options.canShowWhileLocked),
#if HAVE(CORE_ANIMATION_RENDER_SERVER)
        kCAContextIgnoresHitTest : @YES,
        kCAContextDisplayId : @10000
#endif
    };
#if USE(EXTENSIONKIT)
    if (options.useHostable) {
        layerHostingContext->m_hostable = [getBELayerHierarchyClass() layerHierarchyWithOptions:contextOptions error:nil];
        return layerHostingContext;
    }
#endif
    layerHostingContext->m_context = [CAContext remoteContextWithOptions:contextOptions];
#elif !PLATFORM(MACCATALYST)
    [CAContext setAllowsCGSConnections:NO];
    layerHostingContext->m_context = [CAContext remoteContextWithOptions:@{
        kCAContextCIFilterBehavior :  @"ignore",
    }];
#else
    layerHostingContext->m_context = [CAContext contextWithCGSConnection:CGSMainConnectionID() options:@{
        kCAContextCIFilterBehavior : @"ignore",
    }];
#endif
    layerHostingContext->m_cachedContextID = layerHostingContext->contextID();
    return layerHostingContext;
}

#if PLATFORM(MAC)
std::unique_ptr<LayerHostingContext> LayerHostingContext::createForExternalPluginHostingProcess()
{
    auto layerHostingContext = makeUnique<LayerHostingContext>();
    layerHostingContext->m_layerHostingMode = LayerHostingMode::OutOfProcess;
    layerHostingContext->m_context = [CAContext contextWithCGSConnection:CGSMainConnectionID() options:@{ kCAContextCIFilterBehavior : @"ignore" }];
    return layerHostingContext;
}
#endif

std::unique_ptr<LayerHostingContext> LayerHostingContext::createTransportLayerForRemoteHosting(LayerHostingContextID contextID)
{
    auto layerHostingContext = makeUnique<LayerHostingContext>();
    layerHostingContext->m_layerHostingMode = LayerHostingMode::OutOfProcess;
    layerHostingContext->m_cachedContextID = contextID;
    return layerHostingContext;
}

RetainPtr<CALayer> LayerHostingContext::createPlatformLayerForHostingContext(LayerHostingContextID contextID)
{
    return [CALayer _web_renderLayerWithContextID:contextID shouldPreserveFlip:NO];
}

#endif // HAVE(OUT_OF_PROCESS_LAYER_HOSTING)

LayerHostingContext::LayerHostingContext()
{
}

LayerHostingContext::~LayerHostingContext()
{
#if USE(EXTENSIONKIT)
    [m_hostable invalidate];
#endif
}

void LayerHostingContext::setRootLayer(CALayer *rootLayer)
{
#if USE(EXTENSIONKIT)
    if (m_hostable) {
        [m_hostable setLayer:rootLayer];
        return;
    }
#endif
    [m_context setLayer:rootLayer];
}

CALayer *LayerHostingContext::rootLayer() const
{
#if USE(EXTENSIONKIT)
    if (m_hostable)
        return [m_hostable layer];
#endif
    return [m_context layer];
}

LayerHostingContextID LayerHostingContext::contextID() const
{
#if USE(EXTENSIONKIT)
    if (auto xpcDictionary = xpcRepresentation())
        return xpc_dictionary_get_uint64(xpcDictionary.get(), contextIDKey);
#endif
    return [m_context contextId];
}

void LayerHostingContext::invalidate()
{
    [m_context invalidate];
}

void LayerHostingContext::setColorSpace(CGColorSpaceRef colorSpace)
{
    [m_context setColorSpace:colorSpace];
}

CGColorSpaceRef LayerHostingContext::colorSpace() const
{
    return [m_context colorSpace];
}

#if PLATFORM(MAC)
void LayerHostingContext::setColorMatchUntaggedContent(bool colorMatchUntaggedContent)
{
    [m_context setColorMatchUntaggedContent:colorMatchUntaggedContent];
}

bool LayerHostingContext::colorMatchUntaggedContent() const
{
    return [m_context colorMatchUntaggedContent];
}
#endif

void LayerHostingContext::setFencePort(mach_port_t fencePort)
{
#if USE(EXTENSIONKIT)
    ASSERT(!m_hostable);
#endif
    [m_context setFencePort:fencePort];
}

MachSendRight LayerHostingContext::createFencePort()
{
    return MachSendRight::adopt([m_context createFencePort]);
}

void LayerHostingContext::updateCachedContextID(LayerHostingContextID contextID)
{
    m_cachedContextID = contextID;
}

LayerHostingContextID LayerHostingContext::cachedContextID()
{
    return m_cachedContextID;
}

#if USE(EXTENSIONKIT)
OSObjectPtr<xpc_object_t> LayerHostingContext::xpcRepresentation() const
{
    if (!m_hostable)
        return nullptr;
    return [[m_hostable handle] createXPCRepresentation];
}

RetainPtr<BELayerHierarchyHostingTransactionCoordinator> LayerHostingContext::createHostingUpdateCoordinator(mach_port_t sendRight)
{
    auto xpcRepresentation = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_mach_send(xpcRepresentation.get(), machPortKey, sendRight);
    NSError* error = nil;
    auto coordinator = [getBELayerHierarchyHostingTransactionCoordinatorClass() coordinatorWithXPCRepresentation:xpcRepresentation.get() error:&error];
    if (error)
        NSLog(@"Could not create update coordinator, error = %@", error);
    return coordinator;
}

RetainPtr<BELayerHierarchyHandle> LayerHostingContext::createHostingHandle(uint64_t pid, uint64_t contextID)
{
    auto xpcRepresentation = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    xpc_dictionary_set_uint64(xpcRepresentation.get(), processIDKey, pid);
    xpc_dictionary_set_uint64(xpcRepresentation.get(), contextIDKey, contextID);
    NSError* error = nil;
    auto handle = [getBELayerHierarchyHandleClass() handleWithXPCRepresentation:xpcRepresentation.get() error:&error];
    if (error)
        NSLog(@"Could not create layer hierarchy handle, error = %@", error);
    return handle;
}
#endif

} // namespace WebKit
