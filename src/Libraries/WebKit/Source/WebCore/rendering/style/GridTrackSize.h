/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

#include "GridLength.h"

namespace WebCore {

enum GridTrackSizeType {
    LengthTrackSizing,
    MinMaxTrackSizing,
    FitContentTrackSizing
};

// This class represents a <track-size> from the spec. Althought there are 3 different types of
// <track-size> there is always an equivalent minmax() representation that could represent any of
// them. The only special case is fit-content(argument) which is similar to minmax(auto,
// max-content) except that the track size is clamped at argument if it is greater than the auto
// minimum. At the GridTrackSize level we don't need to worry about clamping so we treat that case
// exactly as auto.
//
// We're using a separate attribute to store fit-content argument even though we could directly use
// m_maxTrackBreadth. The reason why we don't do it is because the maxTrackBreadh() call is a hot
// spot, so adding a conditional statement there (to distinguish between fit-content and any other
// case) was causing a severe performance drop.
class GridTrackSize {
public:
    GridTrackSize(const GridLength& length = GridLength({ }), GridTrackSizeType trackSizeType = LengthTrackSizing)
        : m_type(trackSizeType)
        , m_minTrackBreadth(trackSizeType == FitContentTrackSizing ? Length(LengthType::Auto) : length)
        , m_maxTrackBreadth(trackSizeType == FitContentTrackSizing ? Length(LengthType::Auto) : length)
        , m_fitContentTrackBreadth(trackSizeType == FitContentTrackSizing ? length : GridLength(Length(LengthType::Fixed)))
    {
        ASSERT(trackSizeType == LengthTrackSizing || trackSizeType == FitContentTrackSizing);
        ASSERT(trackSizeType != FitContentTrackSizing || length.isLength());
        cacheMinMaxTrackBreadthTypes();
    }

    GridTrackSize(const GridLength& minTrackBreadth, const GridLength& maxTrackBreadth)
        : m_type(MinMaxTrackSizing)
        , m_minTrackBreadth(minTrackBreadth)
        , m_maxTrackBreadth(maxTrackBreadth)
        , m_fitContentTrackBreadth(GridLength(Length(LengthType::Fixed)))
    {
        cacheMinMaxTrackBreadthTypes();
    }

    const GridLength& fitContentTrackBreadth() const
    {
        ASSERT(m_type == FitContentTrackSizing);
        return m_fitContentTrackBreadth;
    }

    const GridLength& minTrackBreadth() const { return m_minTrackBreadth; }
    const GridLength& maxTrackBreadth() const { return m_maxTrackBreadth; }

    GridTrackSizeType type() const { return m_type; }

    bool isContentSized() const { return m_minTrackBreadth.isContentSized() || m_maxTrackBreadth.isContentSized(); }
    bool isFitContent() const { return m_type == FitContentTrackSizing; }

    bool operator==(const GridTrackSize& other) const
    {
        return m_type == other.m_type && m_minTrackBreadth == other.m_minTrackBreadth && m_maxTrackBreadth == other.m_maxTrackBreadth
            && m_fitContentTrackBreadth == other.m_fitContentTrackBreadth;
    }

    void cacheMinMaxTrackBreadthTypes()
    {
        m_minTrackBreadthIsAuto = minTrackBreadth().isLength() && minTrackBreadth().length().isAuto();
        m_minTrackBreadthIsMinContent = minTrackBreadth().isLength() && minTrackBreadth().length().isMinContent();
        m_minTrackBreadthIsMaxContent = minTrackBreadth().isLength() && minTrackBreadth().length().isMaxContent();
        m_maxTrackBreadthIsMaxContent = maxTrackBreadth().isLength() && maxTrackBreadth().length().isMaxContent();
        m_maxTrackBreadthIsMinContent = maxTrackBreadth().isLength() && maxTrackBreadth().length().isMinContent();
        m_maxTrackBreadthIsAuto = maxTrackBreadth().isLength() && maxTrackBreadth().length().isAuto();
        m_maxTrackBreadthIsFixed = maxTrackBreadth().isLength() && maxTrackBreadth().length().isSpecified();

        // These values depend on the above ones so keep them here.
        m_minTrackBreadthIsIntrinsic = m_minTrackBreadthIsMaxContent || m_minTrackBreadthIsMinContent
            || m_minTrackBreadthIsAuto || isFitContent();
        m_maxTrackBreadthIsIntrinsic = m_maxTrackBreadthIsMaxContent || m_maxTrackBreadthIsMinContent
            || m_maxTrackBreadthIsAuto || isFitContent();
    }

    bool hasIntrinsicMinTrackBreadth() const { return m_minTrackBreadthIsIntrinsic; }
    bool hasIntrinsicMaxTrackBreadth() const { return m_maxTrackBreadthIsIntrinsic; }
    bool hasMinOrMaxContentMinTrackBreadth() const { return m_minTrackBreadthIsMaxContent || m_minTrackBreadthIsMinContent; }
    bool hasAutoMinTrackBreadth() const { return m_minTrackBreadthIsAuto; }
    bool hasAutoMaxTrackBreadth() const { return m_maxTrackBreadthIsAuto; }
    bool hasMaxContentMaxTrackBreadth() const { return m_maxTrackBreadthIsMaxContent; }
    bool hasMaxContentOrAutoMaxTrackBreadth() const { return m_maxTrackBreadthIsMaxContent || m_maxTrackBreadthIsAuto; }
    bool hasMinContentMaxTrackBreadth() const { return m_maxTrackBreadthIsMinContent; }
    bool hasMinOrMaxContentMaxTrackBreadth() const { return m_maxTrackBreadthIsMaxContent || m_maxTrackBreadthIsMinContent; }
    bool hasMaxContentMinTrackBreadth() const { return m_minTrackBreadthIsMaxContent; }
    bool hasMinContentMinTrackBreadth() const { return m_minTrackBreadthIsMinContent; }
    bool hasMaxContentMinTrackBreadthAndMaxContentMaxTrackBreadth() const { return m_minTrackBreadthIsMaxContent && m_maxTrackBreadthIsMaxContent; }
    bool hasAutoOrMinContentMinTrackBreadthAndIntrinsicMaxTrackBreadth() const { return (m_minTrackBreadthIsMinContent || m_minTrackBreadthIsAuto) && m_maxTrackBreadthIsIntrinsic; }
    bool hasFixedMaxTrackBreadth() const { return m_maxTrackBreadthIsFixed; }

private:
    GridTrackSizeType m_type;
    GridLength m_minTrackBreadth;
    GridLength m_maxTrackBreadth;
    GridLength m_fitContentTrackBreadth;

    bool m_minTrackBreadthIsAuto : 1;
    bool m_maxTrackBreadthIsAuto : 1;
    bool m_minTrackBreadthIsMaxContent : 1;
    bool m_minTrackBreadthIsMinContent : 1;
    bool m_maxTrackBreadthIsMaxContent : 1;
    bool m_maxTrackBreadthIsMinContent : 1;
    bool m_minTrackBreadthIsIntrinsic : 1;
    bool m_maxTrackBreadthIsIntrinsic : 1;
    bool m_maxTrackBreadthIsFixed : 1;
};

WTF::TextStream& operator<<(WTF::TextStream&, const GridTrackSize&);

} // namespace WebCore
