/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#ifndef DrawErrorUnderline_h
#define DrawErrorUnderline_h

#if USE(CAIRO)

#include <cairo.h>

//
// Draws an error underline that looks like one of:
//
//              H       E                H
//     /\      /\      /\        /\      /\               -
//   A/  \    /  \    /  \     A/  \    /  \              |
//    \   \  /    \  /   /D     \   \  /    \             |
//     \   \/  C   \/   /        \   \/   C  \            | height = heightSquares * square
//      \      /\  F   /          \  F   /\   \           |
//       \    /  \    /            \    /  \   \G         |
//        \  /    \  /              \  /    \  /          |
//         \/      \/                \/      \/           -
//         B                         B
//         |---|
//       unitWidth = (heightSquares - 1) * square
//
// The x, y, width, height passed in give the desired bounding box;
// x/width are adjusted to make the underline a integer number of units
// wide.
//
static inline void drawErrorUnderline(cairo_t* cr, double x, double y, double width, double height)
{
    static const double heightSquares = 2.5;

    double square = height / heightSquares;
    double halfSquare = 0.5 * square;

    double unitWidth = (heightSquares - 1.0) * square;
    int widthUnits = static_cast<int>((width + 0.5 * unitWidth) / unitWidth);

    x += 0.5 * (width - widthUnits * unitWidth);
    width = widthUnits * unitWidth;

    double bottom = y + height;
    double top = y;

    // Bottom of squiggle
    cairo_move_to(cr, x - halfSquare, top + halfSquare); // A

    int i = 0;
    for (i = 0; i < widthUnits; i += 2) {
        double middle = x + (i + 1) * unitWidth;
        double right = x + (i + 2) * unitWidth;

        cairo_line_to(cr, middle, bottom); // B

        if (i + 2 == widthUnits)
            cairo_line_to(cr, right + halfSquare, top + halfSquare); // D
        else if (i + 1 != widthUnits)
            cairo_line_to(cr, right, top + square); // C
    }

    // Top of squiggle
    for (i -= 2; i >= 0; i -= 2) {
        double left = x + i * unitWidth;
        double middle = x + (i + 1) * unitWidth;
        double right = x + (i + 2) * unitWidth;

        if (i + 1 == widthUnits)
            cairo_line_to(cr, middle + halfSquare, bottom - halfSquare); // G
        else {
            if (i + 2 == widthUnits)
                cairo_line_to(cr, right, top); // E

            cairo_line_to(cr, middle, bottom - halfSquare); // F
        }

        cairo_line_to(cr, left, top); // H
    }

    cairo_fill(cr);
}

#endif // USE(CAIRO)

#endif // DrawErrorUnderline_h
