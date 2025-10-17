/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#include "ANGLEUtilitiesCocoa.h"

#if ENABLE(WEBGL)
#include "ANGLEHeaders.h"
#include "ANGLEUtilities.h"
#include "Logging.h"
#include <Metal/Metal.h>
#include <pal/spi/cocoa/MetalSPI.h>
#include <wtf/SoftLinking.h>
#include <wtf/StdLibExtras.h>
#include <wtf/darwin/WeakLinking.h>

#if USE(APPLE_INTERNAL_SDK) && PLATFORM(VISION)
#include <CompositorServices/CompositorServices_Private.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebCore, CompositorServices)

SOFT_LINK_CLASS_FOR_HEADER(WebCore, CP_OBJECT_cp_proxy_process_rasterization_rate_map)
typedef CP_OBJECT_cp_proxy_process_rasterization_rate_map* cp_proxy_process_rasterization_rate_map_t;

SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, CompositorServices, cp_proxy_process_rasterization_rate_map_create, cp_proxy_process_rasterization_rate_map_t, (id<MTLDevice> device, cp_layer_renderer_layout layout, size_t view_count), (device, layout, view_count))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CompositorServices, cp_proxy_process_rasterization_rate_map_create, cp_proxy_process_rasterization_rate_map_t, (id<MTLDevice> device, cp_layer_renderer_layout layout, size_t view_count), (device, layout, view_count))
#define cp_proxy_process_rasterization_rate_map_create softLink_CompositorServices_cp_proxy_process_rasterization_rate_map_create


SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, CompositorServices, cp_rasterization_rate_map_update_shared_from_layered_descriptor, void, (cp_proxy_process_rasterization_rate_map_t proxy_map, MTLRasterizationRateMapDescriptor* descriptor), (proxy_map, descriptor))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CompositorServices, cp_rasterization_rate_map_update_shared_from_layered_descriptor, void, (cp_proxy_process_rasterization_rate_map_t proxy_map, MTLRasterizationRateMapDescriptor* descriptor), (proxy_map, descriptor))
#define cp_rasterization_rate_map_update_shared_from_layered_descriptor softLink_CompositorServices_cp_rasterization_rate_map_update_shared_from_layered_descriptor


SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, CompositorServices, cp_proxy_process_rasterization_rate_map_get_metal_maps, NSArray<id<MTLRasterizationRateMap>>*, (cp_proxy_process_rasterization_rate_map_t proxy_map), (proxy_map))
SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CompositorServices, cp_proxy_process_rasterization_rate_map_get_metal_maps, NSArray<id<MTLRasterizationRateMap>>*, (cp_proxy_process_rasterization_rate_map_t proxy_map), (proxy_map))
#define cp_proxy_process_rasterization_rate_map_get_metal_maps softLink_CompositorServices_cp_proxy_process_rasterization_rate_map_get_metal_maps


SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, CompositorServices, cp_proxy_process_rasterization_rate_map_get_metal_descriptors, NSArray<MTLRasterizationRateMapDescriptor*>*, (cp_proxy_process_rasterization_rate_map_t proxy_map), (proxy_map))
SOFT_LINK_FUNCTION_FOR_HEADER(WebCore, CompositorServices, cp_rasterization_rate_map_update_from_descriptor, void, (cp_proxy_process_rasterization_rate_map_t proxy_map, __unsafe_unretained MTLRasterizationRateMapDescriptor* descriptors[2]), (proxy_map, descriptors))
SOFT_LINK_CLASS_FOR_SOURCE(WebCore, CompositorServices, CP_OBJECT_cp_proxy_process_rasterization_rate_map)

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CompositorServices, cp_drawable_get_layer_renderer_layout, cp_layer_renderer_layout_private, (cp_drawable_t drawable), (drawable))

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CompositorServices, cp_proxy_process_rasterization_rate_map_get_metal_descriptors, NSArray<MTLRasterizationRateMapDescriptor*>*, (cp_proxy_process_rasterization_rate_map_t proxy_map), (proxy_map))

SOFT_LINK_FUNCTION_FOR_SOURCE(WebCore, CompositorServices, cp_rasterization_rate_map_update_from_descriptor, void, (cp_proxy_process_rasterization_rate_map_t proxy_map, __unsafe_unretained MTLRasterizationRateMapDescriptor* descriptors[2]), (proxy_map, descriptors))

#define cp_proxy_process_rasterization_rate_map_get_metal_descriptors softLink_CompositorServices_cp_proxy_process_rasterization_rate_map_get_metal_descriptors
#define cp_proxy_process_rasterization_rate_map_get_metal_descriptors softLink_CompositorServices_cp_proxy_process_rasterization_rate_map_get_metal_descriptors
#define cp_rasterization_rate_map_update_from_descriptor softLink_CompositorServices_cp_rasterization_rate_map_update_from_descriptor
#define cp_drawable_get_layer_renderer_layout softLink_CompositorServices_cp_drawable_get_layer_renderer_layout

#endif

WTF_WEAK_LINK_FORCE_IMPORT(EGL_Initialize);

namespace WebCore {

bool platformIsANGLEAvailable()
{
    // The ANGLE is weak linked in full, and the EGL_Initialize is explicitly weak linked above
    // so that we can detect the case where ANGLE is not present.
    return EGL_Initialize != NULL; // NOLINT
}

void* createPbufferAndAttachIOSurface(GCGLDisplay display, GCGLConfig config, GCGLenum target, GCGLint usageHint, GCGLenum internalFormat, GCGLsizei width, GCGLsizei height, GCGLenum type, IOSurfaceRef surface, GCGLuint plane)
{
    auto eglTextureTarget = target == GL_TEXTURE_RECTANGLE_ANGLE ? EGL_TEXTURE_RECTANGLE_ANGLE : EGL_TEXTURE_2D;

    const EGLint surfaceAttributes[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_IOSURFACE_PLANE_ANGLE, static_cast<EGLint>(plane),
        EGL_TEXTURE_TARGET, static_cast<EGLint>(eglTextureTarget),
        EGL_TEXTURE_INTERNAL_FORMAT_ANGLE, static_cast<EGLint>(internalFormat),
        EGL_TEXTURE_FORMAT, EGL_TEXTURE_RGBA,
        EGL_TEXTURE_TYPE_ANGLE, static_cast<EGLint>(type),
        // Only has an effect on the iOS Simulator.
        EGL_IOSURFACE_USAGE_HINT_ANGLE, usageHint,
        EGL_NONE, EGL_NONE
    };

    EGLSurface pbuffer = EGL_CreatePbufferFromClientBuffer(display, EGL_IOSURFACE_ANGLE, surface, config, surfaceAttributes);
    if (!pbuffer)
        return nullptr;

    if (!EGL_BindTexImage(display, pbuffer, EGL_BACK_BUFFER)) {
        EGL_DestroySurface(display, pbuffer);
        return nullptr;
    }

    return pbuffer;
}

void destroyPbufferAndDetachIOSurface(EGLDisplay display, void* handle)
{
    EGL_ReleaseTexImage(display, handle, EGL_BACK_BUFFER);
    EGL_DestroySurface(display, handle);
}

RetainPtr<id<MTLRasterizationRateMap>> newRasterizationRateMap(GCGLDisplay display, IntSize physicalSizeLeft, IntSize physicalSizeRight, IntSize screenSize, std::span<const float> horizontalSamplesLeft, std::span<const float> verticalSamples, std::span<const float> horizontalSamplesRight)
{
    EGLDeviceEXT device = EGL_NO_DEVICE_EXT;
    if (!EGL_QueryDisplayAttribEXT(display, EGL_DEVICE_EXT, reinterpret_cast<EGLAttrib*>(&device)))
        return nullptr;

    id<MTLDevice> mtlDevice = nil;
    if (!EGL_QueryDeviceAttribEXT(device, EGL_METAL_DEVICE_ANGLE, reinterpret_cast<EGLAttrib*>(&mtlDevice)))
        return nullptr;

    UNUSED_PARAM(physicalSizeLeft);
    UNUSED_PARAM(physicalSizeRight);

#if USE(APPLE_INTERNAL_SDK) && PLATFORM(VISION)
    RetainPtr<MTLRasterizationRateMapDescriptor> descriptor = adoptNS([MTLRasterizationRateMapDescriptor new]);
    id<MTLRasterizationRateMapDescriptorSPI> descriptor_spi = (id<MTLRasterizationRateMapDescriptorSPI>)descriptor.get();
        descriptor_spi.skipSampleValidationAndApplySampleAtTileGranularity = YES;
    descriptor_spi.mutability = MTLMutabilityMutable;
    descriptor_spi.minFactor  = 0.01;

    constexpr MTLSize maxSampleCount { 256, 256, 1 };
    RetainPtr<MTLRasterizationRateLayerDescriptor> layerDescriptorLeft = adoptNS([[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:maxSampleCount]);
    RetainPtr<MTLRasterizationRateLayerDescriptor> layerDescriptorRight = adoptNS([[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:maxSampleCount]);

    if (horizontalSamplesLeft.size() > maxSampleCount.width || horizontalSamplesRight.size() > maxSampleCount.width || verticalSamples.size() > maxSampleCount.height || !layerDescriptorLeft.get() || !layerDescriptorRight.get())
        return nullptr;

    memcpySpan(unsafeMakeSpan([layerDescriptorLeft horizontalSampleStorage], [layerDescriptorLeft sampleCount].width), horizontalSamplesLeft);
    memcpySpan(unsafeMakeSpan([layerDescriptorLeft verticalSampleStorage], [layerDescriptorLeft sampleCount].height), verticalSamples);
    [layerDescriptorLeft setSampleCount:MTLSizeMake(horizontalSamplesLeft.size(), verticalSamples.size(), 0)];

    memcpySpan(unsafeMakeSpan([layerDescriptorRight horizontalSampleStorage], [layerDescriptorRight sampleCount].width), horizontalSamplesRight);
    memcpySpan(unsafeMakeSpan([layerDescriptorRight verticalSampleStorage], [layerDescriptorRight sampleCount].height), verticalSamples);
    [layerDescriptorRight setSampleCount:MTLSizeMake(horizontalSamplesRight.size(), verticalSamples.size(), 0)];

    [descriptor setScreenSize:MTLSizeMake(screenSize.width(), screenSize.height(), 0)];
    [descriptor layers][0] = layerDescriptorLeft.get();
    [descriptor layers][1] = layerDescriptorRight.get();

    auto rateMap = cp_proxy_process_rasterization_rate_map_create(mtlDevice, cp_layer_renderer_layout_shared, 2);
    cp_rasterization_rate_map_update_shared_from_layered_descriptor(rateMap, descriptor.get());

    RetainPtr<id<MTLRasterizationRateMap>> rasterizationRateMap = cp_proxy_process_rasterization_rate_map_get_metal_maps(rateMap).firstObject;
#else
    RetainPtr<id<MTLRasterizationRateMap>> rasterizationRateMap;
    UNUSED_PARAM(display);
    UNUSED_PARAM(physicalSizeLeft);
    UNUSED_PARAM(physicalSizeRight);
    UNUSED_PARAM(screenSize);
    UNUSED_PARAM(horizontalSamplesLeft);
    UNUSED_PARAM(verticalSamples);
    UNUSED_PARAM(horizontalSamplesRight);
#endif
    return rasterizationRateMap;
}

RetainPtr<id<MTLSharedEvent>> newSharedEventWithMachPort(GCGLDisplay display, mach_port_t machPort)
{
    // FIXME: Check for invalid mach_port_t
    EGLDeviceEXT device = EGL_NO_DEVICE_EXT;
    if (!EGL_QueryDisplayAttribEXT(display, EGL_DEVICE_EXT, reinterpret_cast<EGLAttrib*>(&device)))
        return nullptr;

    id<MTLDevice> mtlDevice = nil;
    if (!EGL_QueryDeviceAttribEXT(device, EGL_METAL_DEVICE_ANGLE, reinterpret_cast<EGLAttrib*>(&mtlDevice)))
        return nullptr;

    return adoptNS([(id<MTLDeviceSPI>)mtlDevice newSharedEventWithMachPort:machPort]);
}

RetainPtr<id<MTLSharedEvent>> newSharedEvent(GCGLDisplay display)
{
    EGLDeviceEXT device = EGL_NO_DEVICE_EXT;
    if (!EGL_QueryDisplayAttribEXT(display, EGL_DEVICE_EXT, reinterpret_cast<EGLAttrib*>(&device)))
        return nullptr;

    id<MTLDevice> mtlDevice = nil;
    if (!EGL_QueryDeviceAttribEXT(device, EGL_METAL_DEVICE_ANGLE, reinterpret_cast<EGLAttrib*>(&mtlDevice)))
        return nullptr;

    return adoptNS([mtlDevice newSharedEvent]);
}

}

#endif
