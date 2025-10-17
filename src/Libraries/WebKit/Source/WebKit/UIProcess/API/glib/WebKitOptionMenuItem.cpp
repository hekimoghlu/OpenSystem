/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#include "WebKitOptionMenuItem.h"

#include "WebKitOptionMenuItemPrivate.h"

using namespace WebKit;

/**
 * WebKitOptionMenuItem:
 *
 * One item of a #WebKitOptionMenu.
 *
 * The #WebKitOptionMenu is composed of WebKitOptionMenuItem<!-- -->s.
 * A WebKitOptionMenuItem always has a label and can contain a tooltip text.
 * You can use the WebKitOptionMenuItem of a #WebKitOptionMenu to build your
 * own menus.
 *
 * Since: 2.18
 */

G_DEFINE_BOXED_TYPE(WebKitOptionMenuItem, webkit_option_menu_item, webkit_option_menu_item_copy, webkit_option_menu_item_free)

/**
 * webkit_option_menu_item_copy:
 * @item: a #WebKitOptionMenuItem
 *
 * Make a copy of the #WebKitOptionMenuItem.
 *
 * Returns: (transfer full): A copy of passed in #WebKitOptionMenuItem
 *
 * Since: 2.18
 */
WebKitOptionMenuItem* webkit_option_menu_item_copy(WebKitOptionMenuItem* item)
{
    g_return_val_if_fail(item, nullptr);

    auto* copyItem = static_cast<WebKitOptionMenuItem*>(fastMalloc(sizeof(WebKitOptionMenuItem)));
    new (copyItem) WebKitOptionMenuItem(item);
    return copyItem;
}

/**
 * webkit_option_menu_item_free:
 * @item: A #WebKitOptionMenuItem
 *
 * Free the #WebKitOptionMenuItem.
 *
 * Since: 2.18
 */
void webkit_option_menu_item_free(WebKitOptionMenuItem* item)
{
    g_return_if_fail(item);

    item->~WebKitOptionMenuItem();
    fastFree(item);
}

/**
 * webkit_option_menu_item_get_label:
 * @item: a #WebKitOptionMenuItem
 *
 * Get the label of a #WebKitOptionMenuItem.
 *
 * Returns: The label of @item.
 *
 * Since: 2.18
 */
const gchar* webkit_option_menu_item_get_label(WebKitOptionMenuItem* item)
{
    g_return_val_if_fail(item, nullptr);

    return item->label.data();
}

/**
 * webkit_option_menu_item_get_tooltip:
 * @item: a #WebKitOptionMenuItem
 *
 * Get the tooltip of a #WebKitOptionMenuItem.
 *
 * Returns: The tooltip of @item, or %NULL.
 *
 * Since: 2.18
 */
const gchar* webkit_option_menu_item_get_tooltip(WebKitOptionMenuItem* item)
{
    g_return_val_if_fail(item, nullptr);

    return item->tooltip.isNull() ? nullptr : item->tooltip.data();
}

/**
 * webkit_option_menu_item_is_group_label:
 * @item: a #WebKitOptionMenuItem
 *
 * Whether a #WebKitOptionMenuItem is a group label.
 *
 * Returns: %TRUE if the @item is a group label or %FALSE otherwise.
 *
 * Since: 2.18
 */
gboolean webkit_option_menu_item_is_group_label(WebKitOptionMenuItem* item)
{
    g_return_val_if_fail(item, FALSE);

    return item->isGroupLabel;
}

/**
 * webkit_option_menu_item_is_group_child:
 * @item: a #WebKitOptionMenuItem
 *
 * Whether a #WebKitOptionMenuItem is a group child.
 *
 * Returns: %TRUE if the @item is a group child or %FALSE otherwise.
 *
 * Since: 2.18
 */
gboolean webkit_option_menu_item_is_group_child(WebKitOptionMenuItem* item)
{
    g_return_val_if_fail(item, FALSE);

    return item->isGroupChild;
}

/**
 * webkit_option_menu_item_is_enabled:
 * @item: a #WebKitOptionMenuItem
 *
 * Whether a #WebKitOptionMenuItem is enabled.
 *
 * Returns: %TRUE if the @item is enabled or %FALSE otherwise.
 *
 * Since: 2.18
 */
gboolean webkit_option_menu_item_is_enabled(WebKitOptionMenuItem* item)
{
    g_return_val_if_fail(item, FALSE);

    return item->isEnabled;
}

/**
 * webkit_option_menu_item_is_selected:
 * @item: a #WebKitOptionMenuItem
 *
 * Whether a #WebKitOptionMenuItem is the currently selected one.
 *
 * Returns: %TRUE if the @item is selected or %FALSE otherwise.
 *
 * Since: 2.18
 */
gboolean webkit_option_menu_item_is_selected(WebKitOptionMenuItem* item)
{
    g_return_val_if_fail(item, FALSE);

    return item->isSelected;
}
