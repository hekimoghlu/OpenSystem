/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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

#include "FloatPoint.h"
#include "Path.h"

namespace WebCore {

class PathTraversalState {
public:
    enum class Action {
        TotalLength,
        VectorAtLength,
        SegmentAtLength,
    };

    PathTraversalState(Action, float desiredLength = 0);

public:
    bool processPathElement(PathElement::Type, std::span<const FloatPoint>);
    bool processPathElement(const PathElement& element) { return processPathElement(element.type, element.points); }

    Action action() const { return m_action; }
    void setAction(Action action) { m_action = action; }
    float desiredLength() const { return m_desiredLength; }
    void setDesiredLength(float desiredLength) { m_desiredLength = desiredLength; }

    // Traversing output -- should be read only
    bool success() const { return m_success; }
    float totalLength() const { return m_totalLength; }
    FloatPoint current() const { return m_current; }
    float normalAngle() const { return m_normalAngle; }

private:
    void closeSubpath();
    void moveTo(const FloatPoint&);
    void lineTo(const FloatPoint&);
    void quadraticBezierTo(const FloatPoint&, const FloatPoint&);
    void cubicBezierTo(const FloatPoint&, const FloatPoint&, const FloatPoint&);

    bool finalizeAppendPathElement();
    bool appendPathElement(PathElement::Type, std::span<const FloatPoint>);

private:
    Action m_action;
    bool m_success { false };

    FloatPoint m_current;
    FloatPoint m_start;

    float m_totalLength { 0 };
    float m_desiredLength { 0 };

    // For normal calculations
    FloatPoint m_previous;
    float m_normalAngle { 0 }; // degrees
    bool m_isZeroVector { false };
};

} // namespace WebCore
