/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "AudioBus.h"

#include "AudioFileReader.h"
#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GSpanExtras.h>
#include <wtf/glib/GUniquePtr.h>

#if PLATFORM(GTK)
#define AUDIO_GRESOURCE_PATH "/org/webkitgtk/resources/audio"
#elif PLATFORM(WPE)
#define AUDIO_GRESOURCE_PATH "/org/webkitwpe/resources/audio"
#endif

namespace WebCore {

RefPtr<AudioBus> AudioBus::loadPlatformResource(const char* name, float sampleRate)
{
    GUniquePtr<char> path(g_strdup_printf(AUDIO_GRESOURCE_PATH "/%s", name));
    GRefPtr<GBytes> data = adoptGRef(g_resources_lookup_data(path.get(), G_RESOURCE_LOOKUP_FLAGS_NONE, nullptr));
    ASSERT(data);
    return createBusFromInMemoryAudioFile(span(data), false, sampleRate);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
