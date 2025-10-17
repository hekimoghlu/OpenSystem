/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

#include "GStreamerQuirks.h"

namespace WebCore {

class GStreamerQuirkRealtek final : public GStreamerQuirk {
public:
    GStreamerQuirkRealtek();
    const ASCIILiteral identifier() const final { return "Realtek"_s; }

    GstElement* createWebAudioSink() final;
    void configureElement(GstElement*, const OptionSet<ElementRuntimeCharacteristics>&) final;
    std::optional<bool> isHardwareAccelerated(GstElementFactory*) final;
    Vector<String> disallowedWebAudioDecoders() const final { return m_disallowedWebAudioDecoders; }
    bool shouldParseIncomingLibWebRTCBitStream() const final { return false; }

private:
    Vector<String> m_disallowedWebAudioDecoders;
};

} // namespace WebCore

#endif // USE(GSTREAMER)
