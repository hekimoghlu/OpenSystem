/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
#include "CSSLab.h"

#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSLab);

ExceptionOr<Ref<CSSLab>> CSSLab::create(CSSColorPercent&& lightness, CSSColorNumber&& a, CSSColorNumber&& b, CSSColorPercent&& alpha)
{
    auto rectifiedLightness = rectifyCSSColorPercent(WTFMove(lightness));
    if (rectifiedLightness.hasException())
        return rectifiedLightness.releaseException();
    auto rectifiedA = rectifyCSSColorNumber(WTFMove(a));
    if (rectifiedA.hasException())
        return rectifiedA.releaseException();
    auto rectifiedB = rectifyCSSColorNumber(WTFMove(b));
    if (rectifiedB.hasException())
        return rectifiedB.releaseException();
    auto rectifiedAlpha = rectifyCSSColorPercent(WTFMove(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();

    return adoptRef(*new CSSLab(rectifiedLightness.releaseReturnValue(), rectifiedA.releaseReturnValue(), rectifiedB.releaseReturnValue(), rectifiedAlpha.releaseReturnValue()));
}

CSSLab::CSSLab(RectifiedCSSColorPercent&& lightness, RectifiedCSSColorNumber&& a, RectifiedCSSColorNumber&& b, RectifiedCSSColorPercent&& alpha)
    : m_lightness(WTFMove(lightness))
    , m_a(WTFMove(a))
    , m_b(WTFMove(b))
    , m_alpha(WTFMove(alpha))
{
}

CSSColorPercent CSSLab::l() const
{
    return toCSSColorPercent(m_lightness);
}

ExceptionOr<void> CSSLab::setL(CSSColorPercent&& lightness)
{
    auto rectifiedLightness = rectifyCSSColorPercent(WTFMove(lightness));
    if (rectifiedLightness.hasException())
        return rectifiedLightness.releaseException();
    m_lightness = rectifiedLightness.releaseReturnValue();
    return { };
}

CSSColorNumber CSSLab::a() const
{
    return toCSSColorNumber(m_a);
}

ExceptionOr<void> CSSLab::setA(CSSColorNumber&& a)
{
    auto rectifiedA = rectifyCSSColorNumber(WTFMove(a));
    if (rectifiedA.hasException())
        return rectifiedA.releaseException();
    m_a = rectifiedA.releaseReturnValue();
    return { };
}

CSSColorNumber CSSLab::b() const
{
    return toCSSColorNumber(m_b);
}

ExceptionOr<void> CSSLab::setB(CSSColorNumber&& b)
{
    auto rectifiedB = rectifyCSSColorNumber(WTFMove(b));
    if (rectifiedB.hasException())
        return rectifiedB.releaseException();
    m_b = rectifiedB.releaseReturnValue();
    return { };
}

CSSColorPercent CSSLab::alpha() const
{
    return toCSSColorPercent(m_alpha);
}

ExceptionOr<void> CSSLab::setAlpha(CSSColorPercent&& alpha)
{
    auto rectifiedAlpha = rectifyCSSColorPercent(WTFMove(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();
    m_alpha = rectifiedAlpha.releaseReturnValue();
    return { };
}

} // namespace WebCore
