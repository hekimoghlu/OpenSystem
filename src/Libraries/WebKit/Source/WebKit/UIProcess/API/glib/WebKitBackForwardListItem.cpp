/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include "WebKitBackForwardListItem.h"

#include "WebKitBackForwardListPrivate.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/WTFGType.h>
#include <wtf/text/CString.h>

using namespace WebKit;

/**
 * WebKitBackForwardListItem:
 * @See_also: #WebKitBackForwardList
 *
 * One item of the #WebKitBackForwardList.
 *
 * A history item is part of the #WebKitBackForwardList and consists
 * out of a title and a URI.
 */

struct _WebKitBackForwardListItemPrivate {
    RefPtr<WebBackForwardListItem> webListItem;
    CString uri;
    CString title;
    CString originalURI;
};

WEBKIT_DEFINE_FINAL_TYPE(WebKitBackForwardListItem, webkit_back_forward_list_item, G_TYPE_INITIALLY_UNOWNED, GInitiallyUnowned)

static void webkit_back_forward_list_item_class_init(WebKitBackForwardListItemClass*)
{
}

typedef HashMap<WebBackForwardListItem*, WebKitBackForwardListItem*> HistoryItemsMap;

static HistoryItemsMap& historyItemsMap()
{
    static NeverDestroyed<HistoryItemsMap> itemsMap;
    return itemsMap;
}

static void webkitBackForwardListItemFinalized(gpointer webListItem, GObject* finalizedListItem)
{
    ASSERT_UNUSED(finalizedListItem, G_OBJECT(historyItemsMap().get(static_cast<WebBackForwardListItem*>(webListItem))) == finalizedListItem);
    historyItemsMap().remove(static_cast<WebBackForwardListItem*>(webListItem));
}

WebKitBackForwardListItem* webkitBackForwardListItemGetOrCreate(WebBackForwardListItem* webListItem)
{
    if (!webListItem)
        return 0;

    WebKitBackForwardListItem* listItem = historyItemsMap().get(webListItem);
    if (listItem)
        return listItem;

    listItem = WEBKIT_BACK_FORWARD_LIST_ITEM(g_object_new(WEBKIT_TYPE_BACK_FORWARD_LIST_ITEM, NULL));
    listItem->priv->webListItem = webListItem;

    g_object_weak_ref(G_OBJECT(listItem), webkitBackForwardListItemFinalized, webListItem);
    historyItemsMap().set(webListItem, listItem);

    return listItem;
}

WebBackForwardListItem* webkitBackForwardListItemGetItem(WebKitBackForwardListItem* listItem)
{
    return listItem->priv->webListItem.get();
}

/**
 * webkit_back_forward_list_item_get_uri:
 * @list_item: a #WebKitBackForwardListItem
 *
 * Obtain the URI of the item.
 *
 * This URI may differ from the original URI if the page was,
 * for example, redirected to a new location.
 * See also webkit_back_forward_list_item_get_original_uri().
 *
 * Returns: the URI of @list_item or %NULL
 *    when the URI is empty.
 */
const gchar* webkit_back_forward_list_item_get_uri(WebKitBackForwardListItem* listItem)
{
    g_return_val_if_fail(WEBKIT_IS_BACK_FORWARD_LIST_ITEM(listItem), 0);

    WebKitBackForwardListItemPrivate* priv = listItem->priv;
    String url = priv->webListItem->url();
    if (url.isEmpty())
        return 0;

    priv->uri = url.utf8();
    return priv->uri.data();
}

/**
 * webkit_back_forward_list_item_get_title:
 * @list_item: a #WebKitBackForwardListItem
 *
 * Obtain the title of the item.
 *
 * Returns: the page title of @list_item or %NULL
 *    when the title is empty.
 */
const gchar* webkit_back_forward_list_item_get_title(WebKitBackForwardListItem* listItem)
{
    g_return_val_if_fail(WEBKIT_IS_BACK_FORWARD_LIST_ITEM(listItem), 0);

    WebKitBackForwardListItemPrivate* priv = listItem->priv;
    String title = priv->webListItem->title();
    if (title.isEmpty())
        return 0;

    priv->title = title.utf8();
    return priv->title.data();
}

/**
 * webkit_back_forward_list_item_get_original_uri:
 * @list_item: a #WebKitBackForwardListItem
 *
 * Obtain the original URI of the item.
 *
 * See also webkit_back_forward_list_item_get_uri().
 *
 * Returns: the original URI of @list_item or %NULL
 *    when the original URI is empty.
 */
const gchar* webkit_back_forward_list_item_get_original_uri(WebKitBackForwardListItem* listItem)
{
    g_return_val_if_fail(WEBKIT_IS_BACK_FORWARD_LIST_ITEM(listItem), 0);

    WebKitBackForwardListItemPrivate* priv = listItem->priv;
    String originalURL = priv->webListItem->originalURL();
    if (originalURL.isEmpty())
        return 0;

    priv->originalURI = originalURL.utf8();
    return priv->originalURI.data();
}
