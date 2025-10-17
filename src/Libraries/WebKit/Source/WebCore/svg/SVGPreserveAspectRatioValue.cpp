/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#include "SVGPreserveAspectRatioValue.h"

#include "AffineTransform.h"
#include "FloatRect.h"
#include "SVGParserUtilities.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringParsingBuffer.h>
#include <wtf/text/StringView.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGPreserveAspectRatioValue);

SVGPreserveAspectRatioValue::SVGPreserveAspectRatioValue(StringView value)
{
    parse(value);
}

SVGPreserveAspectRatioValue::SVGPreserveAspectRatioValue(SVGPreserveAspectRatioType align, SVGMeetOrSliceType meetOrSlice)
    : m_align(align)
    , m_meetOrSlice(meetOrSlice)
{
}

ExceptionOr<void> SVGPreserveAspectRatioValue::setAlign(unsigned short align)
{
    if (align == SVG_PRESERVEASPECTRATIO_UNKNOWN || align > SVG_PRESERVEASPECTRATIO_XMAXYMAX)
        return Exception { ExceptionCode::NotSupportedError };

    m_align = static_cast<SVGPreserveAspectRatioType>(align);
    return { };
}

ExceptionOr<void> SVGPreserveAspectRatioValue::setMeetOrSlice(unsigned short meetOrSlice)
{
    if (meetOrSlice == SVG_MEETORSLICE_UNKNOWN || meetOrSlice > SVG_MEETORSLICE_SLICE)
        return Exception { ExceptionCode::NotSupportedError };

    m_meetOrSlice = static_cast<SVGMeetOrSliceType>(meetOrSlice);
    return { };
}

bool SVGPreserveAspectRatioValue::parse(StringView value)
{
    return readCharactersForParsing(value, [&](auto buffer) {
        return parseInternal(buffer, true);
    });
}

bool SVGPreserveAspectRatioValue::parse(StringParsingBuffer<LChar>& buffer, bool validate)
{
    return parseInternal(buffer, validate);
}

bool SVGPreserveAspectRatioValue::parse(StringParsingBuffer<UChar>& buffer, bool validate)
{
    return parseInternal(buffer, validate);
}

template<typename CharacterType> static constexpr std::array<CharacterType, 4> noneDesc  { 'n', 'o', 'n', 'e' };
template<typename CharacterType> static constexpr std::array<CharacterType, 4> meetDesc  { 'm', 'e', 'e', 't' };
template<typename CharacterType> static constexpr std::array<CharacterType, 5> sliceDesc  { 's', 'l', 'i', 'c', 'e' };

template<typename CharacterType> bool SVGPreserveAspectRatioValue::parseInternal(StringParsingBuffer<CharacterType>& buffer, bool validate)
{
    SVGPreserveAspectRatioType align = SVG_PRESERVEASPECTRATIO_XMIDYMID;
    SVGMeetOrSliceType meetOrSlice = SVG_MEETORSLICE_MEET;

    m_align = align;
    m_meetOrSlice = meetOrSlice;

    if (!skipOptionalSVGSpaces(buffer))
        return false;

    if (*buffer == 'n') {
        if (!skipCharactersExactly(buffer, std::span { noneDesc<CharacterType> })) {
            LOG_ERROR("Skipped to parse except for *none* value.");
            return false;
        }
        align = SVG_PRESERVEASPECTRATIO_NONE;
        skipOptionalSVGSpaces(buffer);
    } else if (*buffer == 'x') {
        if (buffer.lengthRemaining() < 8)
            return false;
        if (buffer[1] != 'M' || buffer[4] != 'Y' || buffer[5] != 'M')
            return false;
        if (buffer[2] == 'i') {
            if (buffer[3] == 'n') {
                if (buffer[6] == 'i') {
                    if (buffer[7] == 'n')
                        align = SVG_PRESERVEASPECTRATIO_XMINYMIN;
                    else if (buffer[7] == 'd')
                        align = SVG_PRESERVEASPECTRATIO_XMINYMID;
                    else
                        return false;
                } else if (buffer[6] == 'a' && buffer[7] == 'x')
                    align = SVG_PRESERVEASPECTRATIO_XMINYMAX;
                else
                    return false;
            } else if (buffer[3] == 'd') {
                if (buffer[6] == 'i') {
                    if (buffer[7] == 'n')
                        align = SVG_PRESERVEASPECTRATIO_XMIDYMIN;
                    else if (buffer[7] == 'd')
                        align = SVG_PRESERVEASPECTRATIO_XMIDYMID;
                    else
                        return false;
                } else if (buffer[6] == 'a' && buffer[7] == 'x')
                    align = SVG_PRESERVEASPECTRATIO_XMIDYMAX;
                else
                    return false;
            } else
                return false;
        } else if (buffer[2] == 'a' && buffer[3] == 'x') {
            if (buffer[6] == 'i') {
                if (buffer[7] == 'n')
                    align = SVG_PRESERVEASPECTRATIO_XMAXYMIN;
                else if (buffer[7] == 'd')
                    align = SVG_PRESERVEASPECTRATIO_XMAXYMID;
                else
                    return false;
            } else if (buffer[6] == 'a' && buffer[7] == 'x')
                align = SVG_PRESERVEASPECTRATIO_XMAXYMAX;
            else
                return false;
        } else
            return false;
        buffer += 8;
        skipOptionalSVGSpaces(buffer);
    } else
        return false;

    if (buffer.hasCharactersRemaining()) {
        if (*buffer == 'm') {
            if (!skipCharactersExactly(buffer, std::span { meetDesc<CharacterType> })) {
                LOG_ERROR("Skipped to parse except for *meet* or *slice* value.");
                return false;
            }
            skipOptionalSVGSpaces(buffer);
        } else if (*buffer == 's') {
            if (!skipCharactersExactly(buffer, std::span { sliceDesc<CharacterType> })) {
                LOG_ERROR("Skipped to parse except for *meet* or *slice* value.");
                return false;
            }
            skipOptionalSVGSpaces(buffer);
            if (align != SVG_PRESERVEASPECTRATIO_NONE)
                meetOrSlice = SVG_MEETORSLICE_SLICE;
        }
    }

    if (!buffer.atEnd() && validate)
        return false;

    m_align = align;
    m_meetOrSlice = meetOrSlice;

    return true;
}

void SVGPreserveAspectRatioValue::transformRect(FloatRect& destRect, FloatRect& srcRect) const
{
    if (m_align == SVG_PRESERVEASPECTRATIO_NONE)
        return;

    FloatSize imageSize = srcRect.size();
    float origDestWidth = destRect.width();
    float origDestHeight = destRect.height();
    switch (m_meetOrSlice) {
    case SVGPreserveAspectRatioValue::SVG_MEETORSLICE_UNKNOWN:
        break;
    case SVGPreserveAspectRatioValue::SVG_MEETORSLICE_MEET: {
        float widthToHeightMultiplier = srcRect.height() / srcRect.width();
        if (origDestHeight > origDestWidth * widthToHeightMultiplier) {
            destRect.setHeight(origDestWidth * widthToHeightMultiplier);
            switch (m_align) {
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMINYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMID:
                destRect.setY(destRect.y() + origDestHeight / 2 - destRect.height() / 2);
                break;
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMINYMAX:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMAX:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMAX:
                destRect.setY(destRect.y() + origDestHeight - destRect.height());
                break;
            default:
                break;
            }
        }
        if (origDestWidth > origDestHeight / widthToHeightMultiplier) {
            destRect.setWidth(origDestHeight / widthToHeightMultiplier);
            switch (m_align) {
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMIN:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMAX:
                destRect.setX(destRect.x() + origDestWidth / 2 - destRect.width() / 2);
                break;
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMIN:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMAX:
                destRect.setX(destRect.x() + origDestWidth - destRect.width());
                break;
            default:
                break;
            }
        }
        break;
    }
    case SVGPreserveAspectRatioValue::SVG_MEETORSLICE_SLICE: {
        float widthToHeightMultiplier = srcRect.height() / srcRect.width();
        // if the destination height is less than the height of the image we'll be drawing
        if (origDestHeight < origDestWidth * widthToHeightMultiplier) {
            float destToSrcMultiplier = srcRect.width() / destRect.width();
            srcRect.setHeight(destRect.height() * destToSrcMultiplier);
            switch (m_align) {
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMINYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMID:
                srcRect.setY(srcRect.y() + imageSize.height() / 2 - srcRect.height() / 2);
                break;
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMINYMAX:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMAX:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMAX:
                srcRect.setY(srcRect.y() + imageSize.height() - srcRect.height());
                break;
            default:
                break;
            }
        }
        // if the destination width is less than the width of the image we'll be drawing
        if (origDestWidth < origDestHeight / widthToHeightMultiplier) {
            float destToSrcMultiplier = srcRect.height() / destRect.height();
            srcRect.setWidth(destRect.width() * destToSrcMultiplier);
            switch (m_align) {
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMIN:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMIDYMAX:
                srcRect.setX(srcRect.x() + imageSize.width() / 2 - srcRect.width() / 2);
                break;
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMIN:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMID:
            case SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_XMAXYMAX:
                srcRect.setX(srcRect.x() + imageSize.width() - srcRect.width());
                break;
            default:
                break;
            }
        }
        break;
    }
    }
}

AffineTransform SVGPreserveAspectRatioValue::getCTM(float logicalX, float logicalY, float logicalWidth, float logicalHeight, float physicalWidth, float physicalHeight) const
{
    AffineTransform transform;
    if (!logicalWidth || !logicalHeight || !physicalWidth || !physicalHeight) {
        ASSERT_NOT_REACHED();
        return transform;
    }

    if (m_align == SVG_PRESERVEASPECTRATIO_UNKNOWN)
        return transform;

    double extendedLogicalX = logicalX;
    double extendedLogicalY = logicalY;
    double extendedLogicalWidth = logicalWidth;
    double extendedLogicalHeight = logicalHeight;
    double extendedPhysicalWidth = physicalWidth;
    double extendedPhysicalHeight = physicalHeight;
    double logicalRatio = extendedLogicalWidth / extendedLogicalHeight;
    double physicalRatio = extendedPhysicalWidth / extendedPhysicalHeight;

    if (m_align == SVG_PRESERVEASPECTRATIO_NONE) {
        transform.scaleNonUniform(extendedPhysicalWidth / extendedLogicalWidth, extendedPhysicalHeight / extendedLogicalHeight);
        transform.translate(-extendedLogicalX, -extendedLogicalY);
        return transform;
    }

    if ((logicalRatio < physicalRatio && (m_meetOrSlice == SVG_MEETORSLICE_MEET)) || (logicalRatio >= physicalRatio && (m_meetOrSlice == SVG_MEETORSLICE_SLICE))) {
        transform.scaleNonUniform(extendedPhysicalHeight / extendedLogicalHeight, extendedPhysicalHeight / extendedLogicalHeight);

        if (m_align == SVG_PRESERVEASPECTRATIO_XMINYMIN || m_align == SVG_PRESERVEASPECTRATIO_XMINYMID || m_align == SVG_PRESERVEASPECTRATIO_XMINYMAX)
            transform.translate(-extendedLogicalX, -extendedLogicalY);
        else if (m_align == SVG_PRESERVEASPECTRATIO_XMIDYMIN || m_align == SVG_PRESERVEASPECTRATIO_XMIDYMID || m_align == SVG_PRESERVEASPECTRATIO_XMIDYMAX)
            transform.translate(-extendedLogicalX - (extendedLogicalWidth - extendedPhysicalWidth * extendedLogicalHeight / extendedPhysicalHeight) / 2, -extendedLogicalY);
        else
            transform.translate(-extendedLogicalX - (extendedLogicalWidth - extendedPhysicalWidth * extendedLogicalHeight / extendedPhysicalHeight), -extendedLogicalY);
        
        return transform;
    }

    transform.scaleNonUniform(extendedPhysicalWidth / extendedLogicalWidth, extendedPhysicalWidth / extendedLogicalWidth);

    if (m_align == SVG_PRESERVEASPECTRATIO_XMINYMIN || m_align == SVG_PRESERVEASPECTRATIO_XMIDYMIN || m_align == SVG_PRESERVEASPECTRATIO_XMAXYMIN)
        transform.translate(-extendedLogicalX, -extendedLogicalY);
    else if (m_align == SVG_PRESERVEASPECTRATIO_XMINYMID || m_align == SVG_PRESERVEASPECTRATIO_XMIDYMID || m_align == SVG_PRESERVEASPECTRATIO_XMAXYMID)
        transform.translate(-extendedLogicalX, -extendedLogicalY - (extendedLogicalHeight - extendedPhysicalHeight * extendedLogicalWidth / extendedPhysicalWidth) / 2);
    else
        transform.translate(-extendedLogicalX, -extendedLogicalY - (extendedLogicalHeight - extendedPhysicalHeight * extendedLogicalWidth / extendedPhysicalWidth));

    return transform;
}

String SVGPreserveAspectRatioValue::valueAsString() const
{
    auto alignType = [&]() {
        switch (m_align) {
        case SVG_PRESERVEASPECTRATIO_NONE:
            return "none"_s;
        case SVG_PRESERVEASPECTRATIO_XMINYMIN:
            return "xMinYMin"_s;
        case SVG_PRESERVEASPECTRATIO_XMIDYMIN:
            return "xMidYMin"_s;
        case SVG_PRESERVEASPECTRATIO_XMAXYMIN:
            return "xMaxYMin"_s;
        case SVG_PRESERVEASPECTRATIO_XMINYMID:
            return "xMinYMid"_s;
        case SVG_PRESERVEASPECTRATIO_XMIDYMID:
            return "xMidYMid"_s;
        case SVG_PRESERVEASPECTRATIO_XMAXYMID:
            return "xMaxYMid"_s;
        case SVG_PRESERVEASPECTRATIO_XMINYMAX:
            return "xMinYMax"_s;
        case SVG_PRESERVEASPECTRATIO_XMIDYMAX:
            return "xMidYMax"_s;
        case SVG_PRESERVEASPECTRATIO_XMAXYMAX:
            return "xMaxYMax"_s;
        case SVG_PRESERVEASPECTRATIO_UNKNOWN:
            return "unknown"_s;
        };

        ASSERT_NOT_REACHED();
        return "unknown"_s;
    };

    switch (m_meetOrSlice) {
    default:
    case SVG_MEETORSLICE_UNKNOWN:
        return alignType();
    case SVG_MEETORSLICE_MEET:
        return makeString(alignType(), " meet"_s);
    case SVG_MEETORSLICE_SLICE:
        return makeString(alignType(), " slice"_s);
    }
}

}
