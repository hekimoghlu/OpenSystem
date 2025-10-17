/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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

#if ENABLE(APPLE_PAY)

#include "SystemImage.h"
#include <optional>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum class ApplePayLogoStyle : bool {
    White,
    Black,
};

class WEBCORE_EXPORT ApplePayLogoSystemImage final : public SystemImage {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ApplePayLogoSystemImage, WEBCORE_EXPORT);
public:
    static Ref<ApplePayLogoSystemImage> create(ApplePayLogoStyle applePayLogoStyle)
    {
        return adoptRef(*new ApplePayLogoSystemImage(applePayLogoStyle));
    }

    virtual ~ApplePayLogoSystemImage() = default;

    ApplePayLogoStyle applePayLogoStyle() const { return m_applePayLogoStyle; }

    void draw(GraphicsContext&, const FloatRect&) const final;

private:
    ApplePayLogoSystemImage(ApplePayLogoStyle applePayLogoStyle)
        : SystemImage(SystemImageType::ApplePayLogo)
        , m_applePayLogoStyle(applePayLogoStyle)
    {
    }

    ApplePayLogoStyle m_applePayLogoStyle;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ApplePayLogoSystemImage)
    static bool isType(const WebCore::SystemImage& systemImage) { return systemImage.systemImageType() == WebCore::SystemImageType::ApplePayLogo; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(APPLE_PAY)
