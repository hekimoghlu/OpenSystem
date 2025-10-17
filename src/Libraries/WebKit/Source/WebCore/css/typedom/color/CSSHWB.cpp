/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#include "CSSHWB.h"

#include "Exception.h"
#include "ExceptionOr.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSHWB);

ExceptionOr<Ref<CSSHWB>> CSSHWB::create(Ref<CSSNumericValue>&& hue, CSSNumberish&& whiteness, CSSNumberish&& blackness, CSSNumberish&& alpha)
{
    auto rectifiedHue = rectifyCSSColorAngle(RefPtr { WTFMove(hue) });
    if (rectifiedHue.hasException())
        return rectifiedHue.releaseException();
    auto rectifiedWhiteness = rectifyCSSColorPercent(toCSSColorPercent(whiteness));
    if (rectifiedWhiteness.hasException())
        return rectifiedWhiteness.releaseException();
    auto rectifiedBlackness = rectifyCSSColorPercent(toCSSColorPercent(blackness));
    if (rectifiedBlackness.hasException())
        return rectifiedBlackness.releaseException();
    auto rectifiedAlpha = rectifyCSSColorPercent(toCSSColorPercent(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();
    return adoptRef(*new CSSHWB(std::get<RefPtr<CSSNumericValue>>(rectifiedHue.releaseReturnValue()).releaseNonNull()
        , std::get<RefPtr<CSSNumericValue>>(rectifiedWhiteness.releaseReturnValue()).releaseNonNull()
        , std::get<RefPtr<CSSNumericValue>>(rectifiedBlackness.releaseReturnValue()).releaseNonNull()
        , std::get<RefPtr<CSSNumericValue>>(rectifiedAlpha.releaseReturnValue()).releaseNonNull()));
}

CSSHWB::CSSHWB(Ref<CSSNumericValue>&& hue, Ref<CSSNumericValue>&& whiteness, Ref<CSSNumericValue>&& blackness, Ref<CSSNumericValue>&& alpha)
    : m_hue(WTFMove(hue))
    , m_whiteness(WTFMove(whiteness))
    , m_blackness(WTFMove(blackness))
    , m_alpha(WTFMove(alpha))
{
}

CSSNumericValue& CSSHWB::h() const
{
    return m_hue;
}

ExceptionOr<void> CSSHWB::setH(Ref<CSSNumericValue>&& hue)
{
    auto rectifiedHue = rectifyCSSColorAngle(RefPtr { WTFMove(hue) });
    if (rectifiedHue.hasException())
        return rectifiedHue.releaseException();
    m_hue = std::get<RefPtr<CSSNumericValue>>(rectifiedHue.releaseReturnValue()).releaseNonNull();
    return { };
}

CSSNumberish CSSHWB::w() const
{
    return RefPtr { m_whiteness.copyRef() };
}

ExceptionOr<void> CSSHWB::setW(CSSNumberish&& whiteness)
{
    auto rectifiedWhiteness = rectifyCSSColorPercent(toCSSColorPercent(whiteness));
    if (rectifiedWhiteness.hasException())
        return rectifiedWhiteness.releaseException();
    m_whiteness = std::get<RefPtr<CSSNumericValue>>(rectifiedWhiteness.releaseReturnValue()).releaseNonNull();
    return { };
}

CSSNumberish CSSHWB::b() const
{
    return RefPtr { m_blackness.copyRef() };
}

ExceptionOr<void> CSSHWB::setB(CSSNumberish&& blackness)
{
    auto rectifiedBlackness = rectifyCSSColorPercent(toCSSColorPercent(blackness));
    if (rectifiedBlackness.hasException())
        return rectifiedBlackness.releaseException();
    m_blackness = std::get<RefPtr<CSSNumericValue>>(rectifiedBlackness.releaseReturnValue()).releaseNonNull();
    return { };
}

CSSNumberish CSSHWB::alpha() const
{
    return RefPtr { m_alpha.copyRef() };
}

ExceptionOr<void> CSSHWB::setAlpha(CSSNumberish&& alpha)
{
    auto rectifiedAlpha = rectifyCSSColorPercent(toCSSColorPercent(alpha));
    if (rectifiedAlpha.hasException())
        return rectifiedAlpha.releaseException();
    m_alpha = std::get<RefPtr<CSSNumericValue>>(rectifiedAlpha.releaseReturnValue()).releaseNonNull();
    return { };
}

} // namespace WebCore
