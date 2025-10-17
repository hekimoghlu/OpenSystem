/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)

#include <gst/gst.h>
#include <wtf/Forward.h>

namespace WebCore {
class CaptureDevice;
};

typedef struct _GStreamerMockDevice GStreamerMockDevice;
typedef struct _GStreamerMockDeviceClass GStreamerMockDeviceClass;
typedef struct _GStreamerMockDevicePrivate GStreamerMockDevicePrivate;

#define GST_TYPE_MOCK_DEVICE                 (webkit_mock_device_get_type())
#define GST_IS_MOCK_DEVICE(obj)              (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_MOCK_DEVICE))
#define GST_IS_MOCK_DEVICE_CLASS(klass)      (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_MOCK_DEVICE))
#define WEBKIT_MOCK_DEVICE_GET_CLASS(obj)    (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_MOCK_DEVICE, GStreamerMockDeviceClass))
#define WEBKIT_MOCK_DEVICE(obj)              (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_MOCK_DEVICE, GStreamerMockDevice))
#define WEBKIT_MOCK_DEVICE_CLASS(klass)      (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_DEVICE, GStreamerMockDeviceClass))
#define WEBKIT_MOCK_DEVICE_CAST(obj)         ((GStreamerMockDevice*)(obj))

struct _GStreamerMockDevice {
    GstDevice parent;
    GStreamerMockDevicePrivate* priv;
};

struct _GStreamerMockDeviceClass {
    GstDeviceClass parentClass;
};

GType webkit_mock_device_get_type(void);

GstDevice* webkitMockDeviceCreate(const WebCore::CaptureDevice&);

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
