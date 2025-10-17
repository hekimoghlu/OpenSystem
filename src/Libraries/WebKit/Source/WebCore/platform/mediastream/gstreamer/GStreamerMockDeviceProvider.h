/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

typedef struct _GStreamerMockDeviceProvider GStreamerMockDeviceProvider;
typedef struct _GStreamerMockDeviceProviderClass GStreamerMockDeviceProviderClass;
typedef struct _GStreamerMockDeviceProviderPrivate GStreamerMockDeviceProviderPrivate;

#define GST_TYPE_MOCK_DEVICE_PROVIDER                 (webkit_mock_device_provider_get_type())
#define GST_IS_MOCK_DEVICE_PROVIDER(obj)              (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_MOCK_DEVICE_PROVIDER))
#define GST_IS_MOCK_DEVICE_PROVIDER_CLASS(klass)      (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_MOCK_DEVICE_PROVIDER))
#define WEBKIT_MOCK_DEVICE_PROVIDER_GET_CLASS(obj)       (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_MOCK_DEVICE_PROVIDER, GStreamerMockDeviceProviderClass))
#define WEBKIT_MOCK_DEVICE_PROVIDER(obj)                 (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_MOCK_DEVICE_PROVIDER, GStreamerMockDeviceProvider))
#define WEBKIT_MOCK_DEVICE_PROVIDER_CLASS(klass)         (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_DEVICE_PROVIDER, GStreamerMockDeviceProviderClass))
#define WEBKIT_MOCK_DEVICE_PROVIDER_CAST(obj)            ((GStreamerMockDeviceProvider *)(obj))

struct _GStreamerMockDeviceProvider {
    GstDeviceProvider parent;
    GStreamerMockDeviceProviderPrivate* priv;
};

struct _GStreamerMockDeviceProviderClass {
    GstDeviceProviderClass parentClass;
};

GType webkit_mock_device_provider_get_type(void);

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
