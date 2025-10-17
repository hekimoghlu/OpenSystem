/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#include "WPERectangle.h"

#include <wtf/FastMalloc.h>

/**
 * WPERectangle:
 * @x: The X coordinate of the top-left corner of the rectangle.
 * @y: The Y coordinate of the top-left corner of the rectangle.
 * @width: The width of the rectangle.
 * @height: The height of the rectangle.
 *
 * Boxed type representing a rectangle with integer coordinates.
 */

static WPERectangle* wpe_rectangle_copy(WPERectangle* rectangle)
{
    g_return_val_if_fail(rectangle, nullptr);

    return static_cast<WPERectangle*>(fastMemDup(rectangle, sizeof(WPERectangle)));
}

static void wpe_rectangle_free(WPERectangle* rectangle)
{
    g_return_if_fail(rectangle);

    fastFree(rectangle);
}

G_DEFINE_BOXED_TYPE(WPERectangle, wpe_rectangle, wpe_rectangle_copy, wpe_rectangle_free)
