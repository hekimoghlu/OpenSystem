/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#include "GStreamerMockDeviceProvider.h"

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)

#include "GStreamerMockDevice.h"
#include "MockRealtimeMediaSourceCenter.h"
#include <wtf/glib/WTFGType.h>

using namespace WebCore;

struct _GStreamerMockDeviceProviderPrivate {
};

GST_DEBUG_CATEGORY_STATIC(webkitGstMockDeviceProviderDebug);
#define GST_CAT_DEFAULT webkitGstMockDeviceProviderDebug

WEBKIT_DEFINE_TYPE_WITH_CODE(GStreamerMockDeviceProvider, webkit_mock_device_provider, GST_TYPE_DEVICE_PROVIDER, GST_DEBUG_CATEGORY_INIT(webkitGstMockDeviceProviderDebug, "webkitmockdeviceprovider", 0, "Mock Device Provider"))

static GList* webkitMockDeviceProviderProbe([[maybe_unused]] GstDeviceProvider* provider)
{
    if (!MockRealtimeMediaSourceCenter::mockRealtimeMediaSourceCenterEnabled()) {
        GST_INFO_OBJECT(provider, "Mock capture sources are disabled, returning empty device list");
        return nullptr;
    }

    GST_INFO_OBJECT(provider, "Probing");
    GList* devices = nullptr;
    auto& sourceCenter = MockRealtimeMediaSourceCenter::singleton();
    for (auto& device : sourceCenter.videoDevices())
        devices = g_list_prepend(devices, webkitMockDeviceCreate(device));
    for (auto& device : sourceCenter.microphoneDevices())
        devices = g_list_prepend(devices, webkitMockDeviceCreate(device));
    for (auto& device : sourceCenter.displayDevices())
        devices = g_list_prepend(devices, webkitMockDeviceCreate(device));
    devices = g_list_reverse(devices);
    return devices;
}

static void webkit_mock_device_provider_class_init(GStreamerMockDeviceProviderClass* klass)
{
    auto* providerClass = GST_DEVICE_PROVIDER_CLASS(klass);

    providerClass->probe = GST_DEBUG_FUNCPTR(webkitMockDeviceProviderProbe);

    gst_device_provider_class_set_static_metadata(providerClass, "WebKit Mock Device Provider", "Source/Audio/Video",
        "List and provide WebKit mock source devices", "Philippe Normand <philn@igalia.com>");
}

#undef GST_CAT_DEFAULT

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
