/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#if USE(APPKIT)

#include "AppKitControlSystemImage.h"
#include <optional>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WEBCORE_EXPORT ScrollbarTrackCornerSystemImageMac final : public AppKitControlSystemImage {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ScrollbarTrackCornerSystemImageMac, WEBCORE_EXPORT);
public:
    static Ref<ScrollbarTrackCornerSystemImageMac> create();
    static Ref<ScrollbarTrackCornerSystemImageMac> create(WebCore::Color&& tintColor, bool useDarkAppearance);

    virtual ~ScrollbarTrackCornerSystemImageMac() = default;

    void drawControl(GraphicsContext&, const FloatRect&) const final;

private:
    ScrollbarTrackCornerSystemImageMac();
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ScrollbarTrackCornerSystemImageMac)
    static bool isType(const WebCore::AppKitControlSystemImage& systemImage) { return systemImage.controlType() == WebCore::AppKitControlSystemImageType::ScrollbarTrackCorner; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(APPKIT)
