/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

#include "Color.h"
#include "SystemImage.h"
#include <optional>
#include <wtf/ArgumentCoder.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Color;

enum class AppKitControlSystemImageType : uint8_t {
    ScrollbarTrackCorner,
};

class WEBCORE_EXPORT AppKitControlSystemImage : public SystemImage {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(AppKitControlSystemImage, WEBCORE_EXPORT);
public:
    virtual ~AppKitControlSystemImage() = default;

    void draw(GraphicsContext&, const FloatRect&) const final;

    virtual void drawControl(GraphicsContext&, const FloatRect&) const { }

    AppKitControlSystemImageType controlType() const { return m_controlType; }

    Color tintColor() const { return m_tintColor; }
    void setTintColor(const Color& tintColor) { m_tintColor = tintColor; }

    bool useDarkAppearance() const { return m_useDarkAppearance; }
    void setUseDarkAppearance(bool useDarkAppearance) { m_useDarkAppearance = useDarkAppearance; }

protected:
    AppKitControlSystemImage(AppKitControlSystemImageType);

private:
    friend struct IPC::ArgumentCoder<AppKitControlSystemImage, void>;
    AppKitControlSystemImageType m_controlType;

    Color m_tintColor;
    bool m_useDarkAppearance { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AppKitControlSystemImage)
    static bool isType(const WebCore::SystemImage& systemImage) { return systemImage.systemImageType() == WebCore::SystemImageType::AppKitControl; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(APPKIT)
