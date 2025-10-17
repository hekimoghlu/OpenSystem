/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
#include "WPEColor.h"

/**
 * WPEColor:
 * @red: Red channel, between 0.0 and 1.0 inclusive
 * @green: Green channel, between 0.0 and 1.0 inclusive
 * @blue: Blue channel, between 0.0 and 1.0 inclusive
 * @alpha: Alpha channel, between 0.0 and 1.0 inclusive
 *
 * Boxed type representing a RGBA color.
 */

/**
 * wpe_color_copy:
 * @color: a #WPEColor
 *
 * Make a copy of @color.
 *
 * Returns: (transfer full): A copy of passed in #WPEColor.
 */
WPEColor* wpe_color_copy(WPEColor* color)
{
    g_return_val_if_fail(color, nullptr);

    WPEColor* copy = static_cast<WPEColor*>(fastZeroedMalloc(sizeof(WPEColor)));
    copy->red = color->red;
    copy->green = color->green;
    copy->blue = color->blue;
    copy->alpha = color->alpha;
    return copy;
}

/**
 * wpe_color_free:
 * @color: a #WPEColor
 *
 * Free the #WPEColor.
 */
void wpe_color_free(WPEColor* color)
{
    g_return_if_fail(color);

    fastFree(color);
}

G_DEFINE_BOXED_TYPE(WPEColor, wpe_color, wpe_color_copy, wpe_color_free)
