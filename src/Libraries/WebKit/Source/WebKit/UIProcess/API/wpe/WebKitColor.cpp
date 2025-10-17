/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#include "WebKitColor.h"

#include "WebKitColorPrivate.h"
#include <WebCore/CSSParser.h>

/**
 * WebKitColor:
 * @red: Red channel, between 0.0 and 1.0 inclusive
 * @green: Green channel, between 0.0 and 1.0 inclusive
 * @blue: Blue channel, between 0.0 and 1.0 inclusive
 * @alpha: Alpha channel, between 0.0 and 1.0 inclusive
 *
 * Boxed type representing a RGBA color.
 *
 * Since: 2.24
 */

/**
 * webkit_color_copy:
 * @color: a #WebKitColor
 *
 * Make a copy of @color.
 *
 * Returns: (transfer full): A copy of passed in #WebKitColor.
 *
 * Since: 2.24
 */
WebKitColor* webkit_color_copy(WebKitColor* color)
{
    g_return_val_if_fail(color, nullptr);

    WebKitColor* copy = static_cast<WebKitColor*>(fastZeroedMalloc(sizeof(WebKitColor)));
    copy->red = color->red;
    copy->green = color->green;
    copy->blue = color->blue;
    copy->alpha = color->alpha;
    return copy;
}

/**
 * webkit_color_free:
 * @color: a #WebKitColor
 *
 * Free the #WebKitColor.
 *
 * Since: 2.24
 */
void webkit_color_free(WebKitColor* color)
{
    g_return_if_fail(color);

    fastFree(color);
}

G_DEFINE_BOXED_TYPE(WebKitColor, webkit_color, webkit_color_copy, webkit_color_free);

const WebCore::Color webkitColorToWebCoreColor(WebKitColor* color)
{
    return WebCore::convertColor<WebCore::SRGBA<uint8_t>>(WebCore::SRGBA<float> { static_cast<float>(color->red), static_cast<float>(color->green), static_cast<float>(color->blue), static_cast<float>(color->alpha) });
}

void webkitColorFillFromWebCoreColor(const WebCore::Color& webCoreColor, WebKitColor* color)
{
    RELEASE_ASSERT(webCoreColor.isValid());

    auto [r, g, b, a] = webCoreColor.toColorTypeLossy<WebCore::SRGBA<float>>().resolved();
    color->red = r;
    color->green = g;
    color->blue = b;
    color->alpha = a;
}

/**
 * webkit_color_parse:
 * @color: a #WebKitColor to fill in
 * @color_string: color representation as color nickname or HEX string
 *
 * Create a new #WebKitColor for the given @color_string
 * representation. There are two valid representation types: standard color
 * names (see https://htmlcolorcodes.com/color-names/ for instance) or HEX
 * values.
 *
 * Returns: a #gboolean indicating if the @color was correctly filled in or not.
 *
 * Since: 2.24
 */
gboolean webkit_color_parse(WebKitColor* color, const gchar* colorString)
{
    g_return_val_if_fail(color, FALSE);
    g_return_val_if_fail(colorString, FALSE);

    auto webCoreColor = WebCore::CSSParser::parseColorWithoutContext(String::fromLatin1(colorString));
    if (!webCoreColor.isValid())
        return FALSE;

    webkitColorFillFromWebCoreColor(webCoreColor, color);
    return TRUE;
}
