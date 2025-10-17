/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
#include "CSSLCH.h"

#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSLCH);

ExceptionOr<Ref<CSSLCH>> CSSLCH::create(CSSColorPercent&& lightness, CSSColorPercent&& chroma, CSSColorAngle&& hue, CSSColorPercent&& alpha)
{
    auto rectifiedLightness = rectifyCSSColorPercent(WTFMove(lightness));
    if (rectifiedLightness.hasException())
        return rectifiedLightness.releaseException();
    auto rectifiedChroma = rectifyCSSColorPercent(WTFMove(chroma));
    if (rectifiedChroma.hasException())
        return rectifiedChroma.releaseException();
    auto rectifiedHue = rectifyCSSColorAngle(WTFMove(hue));
    if (rectifiedHue.hasException())
        return rectifiedHue.releaseException();
    auto rectifiedAlpha = rectifyCSSColorPercent(WTFMove(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();

    return adoptRef(*new CSSLCH(rectifiedLightness.releaseReturnValue(), rectifiedChroma.releaseReturnValue(), rectifiedHue.releaseReturnValue(), rectifiedAlpha.releaseReturnValue()));

}

CSSLCH::CSSLCH(RectifiedCSSColorPercent&& lightness, RectifiedCSSColorPercent&& chroma, RectifiedCSSColorAngle&& hue, RectifiedCSSColorPercent&& alpha)
    : m_lightness(WTFMove(lightness))
    , m_chroma(WTFMove(chroma))
    , m_hue(WTFMove(hue))
    , m_alpha(WTFMove(alpha))
{
}

CSSColorPercent CSSLCH::l() const
{
    return toCSSColorAngle(m_lightness);
}

ExceptionOr<void> CSSLCH::setL(CSSColorPercent&& lightness)
{
    auto rectifiedLightness = rectifyCSSColorPercent(WTFMove(lightness));
    if (rectifiedLightness.hasException())
        return rectifiedLightness.releaseException();
    m_lightness = rectifiedLightness.releaseReturnValue();
    return { };
}

CSSColorPercent CSSLCH::c() const
{
    return toCSSColorAngle(m_chroma);
}

ExceptionOr<void> CSSLCH::setC(CSSColorPercent&& chroma)
{
    auto rectifiedChroma = rectifyCSSColorPercent(WTFMove(chroma));
    if (rectifiedChroma.hasException())
        return rectifiedChroma.releaseException();
    m_chroma = rectifiedChroma.releaseReturnValue();
    return { };
}

CSSColorAngle CSSLCH::h() const
{
    return toCSSColorAngle(m_hue);
}

ExceptionOr<void> CSSLCH::setH(CSSColorAngle&& hue)
{
    auto rectifiedHue = rectifyCSSColorAngle(WTFMove(hue));
    if (rectifiedHue.hasException())
        return rectifiedHue.releaseException();
    m_hue = rectifiedHue.releaseReturnValue();
    return { };
}

CSSColorPercent CSSLCH::alpha() const
{
    return toCSSColorPercent(m_alpha);
}

ExceptionOr<void> CSSLCH::setAlpha(CSSColorPercent&& alpha)
{
    auto rectifiedAlpha = rectifyCSSColorPercent(WTFMove(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();
    m_alpha = rectifiedAlpha.releaseReturnValue();
    return { };
}

} // namespace WebCore
