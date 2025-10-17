/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
#include "WebKitInputMethodContext.h"

#include "WebKitInputMethodContextPrivate.h"
#include <WebCore/Color.h>

using namespace WebCore;

/**
 * webkit_input_method_underline_set_color:
 * @underline: a #WebKitInputMethodUnderline
 * @rgba: (nullable): a #GdkRGBA or %NULL
 *
 * Set the color of the underline.
 *
 * If @rgba is %NULL the foreground text color will be used
 * for the underline too.
 *
 * Since: 2.28
 */
void webkit_input_method_underline_set_color(WebKitInputMethodUnderline* underline, const GdkRGBA* rgba)
{
    g_return_if_fail(underline);

    if (!rgba) {
        underline->underline.compositionUnderlineColor = CompositionUnderlineColor::TextColor;
        return;
    }

    underline->underline.compositionUnderlineColor = CompositionUnderlineColor::GivenColor;
    underline->underline.color = Color(*rgba);
}

/**
 * webkit_input_method_context_filter_key_event:
 * @context: a #WebKitInputMethodContext
 * @key_event: the key event to filter
 *
 * Allow @key_event to be handled by the input method.
 *
 * If %TRUE is returned, then no further processing should be
 * done for the key event.
 *
 * Returns: %TRUE if the key event was handled, or %FALSE otherwise
 *
 * Since: 2.28
 */
#if USE(GTK4)
gboolean webkit_input_method_context_filter_key_event(WebKitInputMethodContext* context, GdkEvent* keyEvent)
#else
gboolean webkit_input_method_context_filter_key_event(WebKitInputMethodContext* context, GdkEventKey* keyEvent)
#endif
{
    g_return_val_if_fail(WEBKIT_IS_INPUT_METHOD_CONTEXT(context), FALSE);
    g_return_val_if_fail(keyEvent, FALSE);

    auto* imClass = WEBKIT_INPUT_METHOD_CONTEXT_GET_CLASS(context);
    return imClass->filter_key_event ? imClass->filter_key_event(context, keyEvent) : FALSE;
}
