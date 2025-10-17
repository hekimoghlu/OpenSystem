/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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

#if USE(GSTREAMER)

#include "GStreamerQuirkBroadcomBase.h"
#include "GStreamerQuirks.h"

namespace WebCore {

class GStreamerQuirkBcmNexus final : public GStreamerQuirkBroadcomBase {
public:
    GStreamerQuirkBcmNexus();
    const ASCIILiteral identifier() const final { return "BcmNexus"_s; }

    std::optional<bool> isHardwareAccelerated(GstElementFactory*) final;
    std::optional<GstElementFactoryListType> audioVideoDecoderFactoryListType() const final { return GST_ELEMENT_FACTORY_TYPE_PARSER; }
    Vector<String> disallowedWebAudioDecoders() const final { return m_disallowedWebAudioDecoders; }

private:
    Vector<String> m_disallowedWebAudioDecoders;
};

} // namespace WebCore

#endif // USE(GSTREAMER)
