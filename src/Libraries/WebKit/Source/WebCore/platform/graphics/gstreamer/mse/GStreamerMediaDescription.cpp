/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#include "GStreamerMediaDescription.h"
#include "GStreamerCommon.h"

#include <gst/pbutils/pbutils.h>
#include <wtf/text/MakeString.h>

#if ENABLE(VIDEO) && USE(GSTREAMER) && ENABLE(MEDIA_SOURCE)

namespace WebCore {

bool GStreamerMediaDescription::isVideo() const
{
    return doCapsHaveType(m_caps.get(), GST_VIDEO_CAPS_TYPE_PREFIX);
}

bool GStreamerMediaDescription::isAudio() const
{
    return doCapsHaveType(m_caps.get(), GST_AUDIO_CAPS_TYPE_PREFIX);
}

bool GStreamerMediaDescription::isText() const
{
    // FIXME: Implement proper text track support.
    return false;
}

String GStreamerMediaDescription::extractCodecName(const GRefPtr<GstCaps>& caps) const
{
    GRefPtr<GstCaps> originalCaps = caps;

    if (areEncryptedCaps(originalCaps.get())) {
        originalCaps = adoptGRef(gst_caps_copy(originalCaps.get()));
        GstStructure* structure = gst_caps_get_structure(originalCaps.get(), 0);

        if (!gst_structure_has_field(structure, "original-media-type"))
            return String();

        auto originalMediaType = WebCore::gstStructureGetString(structure, "original-media-type"_s);
        RELEASE_ASSERT(originalMediaType);
        gst_structure_set_name(structure, originalMediaType.toStringWithoutCopying().ascii().data());

        // Remove the DRM related fields from the caps.
        gstStructureFilterAndMapInPlace(structure, [](GstId id, GValue*) -> bool {
            auto idView = gstIdToString(id);
            if (idView.startsWith("protection-system"_s) || idView.startsWith("original-media-type"_s))
                return false;
            return true;
        });
    }

    GUniquePtr<gchar> description(gst_pb_utils_get_codec_description(originalCaps.get()));
    auto codecName = String::fromLatin1(description.get());

    // Report "H.264 (Main Profile)" and "H.264 (High Profile)" just as "H.264" to allow changes between both variants
    // go unnoticed to the SourceBuffer layer.
    if (codecName.startsWith("H.264"_s)) {
        size_t braceStart = codecName.find(" ("_s);
        size_t braceEnd = codecName.find(')', braceStart + 1);
        if (braceStart != notFound && braceEnd != notFound) {
            StringView codecNameView { codecName };
            codecName = makeString(codecNameView.left(braceStart), codecNameView.substring(braceEnd + 1));
        }
    }

    return codecName;
}

} // namespace WebCore.

#endif // USE(GSTREAMER)
