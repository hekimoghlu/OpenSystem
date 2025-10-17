/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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

#include "Color.h"
#include "FloatPoint.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class FloatSize;
class GraphicsContext;
class Path;

class InspectorOverlayLabel {
    WTF_MAKE_TZONE_ALLOCATED(InspectorOverlayLabel);
public:
    struct Arrow {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        enum class Direction : uint8_t {
            None,
            Down,
            Up,
            Left,
            Right,
        };

        enum class Alignment : uint8_t {
            None,
            Leading, // Positioned at the left/top side of edge.
            Middle, // Positioned at the center on the edge.
            Trailing, // Positioned at the right/bottom side of the edge.
        };

        Direction direction;
        Alignment alignment;

        Arrow(Direction direction, Alignment alignment)
            : direction(direction)
            , alignment(alignment)
        {
            ASSERT(alignment != Alignment::None || direction == Direction::None);
        }
    };

    struct Content {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        struct Decoration {
            enum class Type : uint8_t {
                None,
                Bordered,
            };

            Type type;
            Color color;
        };

        String text;
        Color textColor;
        Decoration decoration { Decoration::Type::None, Color::transparentBlack };
    };

    WEBCORE_EXPORT InspectorOverlayLabel(Vector<Content>&&, FloatPoint, Color backgroundColor, Arrow);
    InspectorOverlayLabel(const String&, FloatPoint, Color backgroundColor, Arrow);

    Path draw(GraphicsContext&, float maximumLineWidth = 0);

    static FloatSize expectedSize(const Vector<Content>&, Arrow::Direction);
    static FloatSize expectedSize(const String&, Arrow::Direction);

private:
    friend struct IPC::ArgumentCoder<InspectorOverlayLabel, void>;
    Vector<Content> m_contents;
    FloatPoint m_location;
    Color m_backgroundColor;
    Arrow m_arrow;
};

} // namespace WebCore
