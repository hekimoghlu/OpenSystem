/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
#include "GStreamerMockDevice.h"

#include "CaptureDevice.h"
#include "GStreamerCommon.h"
#include <wtf/glib/WTFGType.h>

using namespace WebCore;

struct _GStreamerMockDevicePrivate {
};

GST_DEBUG_CATEGORY_STATIC(webkitGstMockDeviceDebug);
#define GST_CAT_DEFAULT webkitGstMockDeviceDebug

WEBKIT_DEFINE_TYPE_WITH_CODE(GStreamerMockDevice, webkit_mock_device, GST_TYPE_DEVICE, GST_DEBUG_CATEGORY_INIT(webkitGstMockDeviceDebug, "webkitmockdevice", 0, "Mock Device"))

static GstElement* webkitMockDeviceCreateElement([[maybe_unused]] GstDevice* device, const char* name)
{
    GST_INFO_OBJECT(device, "Creating source element for device %s", name);
    auto* element = makeGStreamerElement("appsrc", name);
    g_object_set(element, "format", GST_FORMAT_TIME, "is-live", TRUE, "do-timestamp", TRUE, nullptr);
    return element;
}

static void webkit_mock_device_class_init(GStreamerMockDeviceClass* klass)
{
    auto* deviceClass = GST_DEVICE_CLASS(klass);
    deviceClass->create_element = GST_DEBUG_FUNCPTR(webkitMockDeviceCreateElement);
}

GstDevice* webkitMockDeviceCreate(const CaptureDevice& captureDevice)
{
    const char* deviceClass;
    GRefPtr<GstCaps> caps;

    switch (captureDevice.type()) {
    case CaptureDevice::DeviceType::Camera:
    case CaptureDevice::DeviceType::Screen:
    case CaptureDevice::DeviceType::Window:
        deviceClass = "Video/Source";
        caps = adoptGRef(gst_caps_new_empty_simple("video/x-raw"));
        break;
    case CaptureDevice::DeviceType::Microphone:
        deviceClass = "Audio/Source";
        caps = adoptGRef(gst_caps_new_empty_simple("audio/x-raw"));
        break;
    default:
        deviceClass = "unknown/unknown";
        caps = adoptGRef(gst_caps_new_any());
        break;
    }

    auto displayName = captureDevice.label();
    GUniquePtr<GstStructure> properties(gst_structure_new("webkit-mock-device", "persistent-id", G_TYPE_STRING, captureDevice.persistentId().ascii().data(), "is-default", G_TYPE_BOOLEAN, captureDevice.isDefault(), nullptr));
    auto* device = WEBKIT_MOCK_DEVICE_CAST(g_object_new(GST_TYPE_MOCK_DEVICE, "display-name", displayName.ascii().data(), "device-class", deviceClass, "caps", caps.get(), "properties", properties.get(), nullptr));
    gst_object_ref_sink(device);
    return GST_DEVICE_CAST(device);
}

#undef GST_CAT_DEFAULT

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
