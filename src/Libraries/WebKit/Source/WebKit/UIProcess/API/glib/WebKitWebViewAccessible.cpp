/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "WebKitWebViewAccessible.h"

#if USE(ATK)

#include <wtf/glib/WTFGType.h>

struct _WebKitWebViewAccessiblePrivate {
    gpointer webView;
};

WEBKIT_DEFINE_TYPE(WebKitWebViewAccessible, webkit_web_view_accessible, ATK_TYPE_SOCKET)

static void webkitWebViewAccessibleInitialize(AtkObject* atkObject, gpointer data)
{
    if (ATK_OBJECT_CLASS(webkit_web_view_accessible_parent_class)->initialize)
        ATK_OBJECT_CLASS(webkit_web_view_accessible_parent_class)->initialize(atkObject, data);

    webkitWebViewAccessibleSetWebView(WEBKIT_WEB_VIEW_ACCESSIBLE(atkObject), data);
    atk_object_set_role(atkObject, ATK_ROLE_FILLER);
}

static AtkStateSet* webkitWebViewAccessibleRefStateSet(AtkObject* atkObject)
{
    WebKitWebViewAccessible* accessible = WEBKIT_WEB_VIEW_ACCESSIBLE(atkObject);

    AtkStateSet* stateSet;
    if (accessible->priv->webView) {
        // Use the implementation of AtkSocket if the web view is still alive.
        stateSet = ATK_OBJECT_CLASS(webkit_web_view_accessible_parent_class)->ref_state_set(atkObject);
        if (!atk_socket_is_occupied(ATK_SOCKET(atkObject)))
            atk_state_set_add_state(stateSet, ATK_STATE_TRANSIENT);
    } else {
        // If the web view is no longer alive, save some remote calls
        // (because of AtkSocket's implementation of ref_state_set())
        // and just return that this AtkObject is defunct.
        stateSet = atk_state_set_new();
        atk_state_set_add_state(stateSet, ATK_STATE_DEFUNCT);
    }

    return stateSet;
}

static gint webkitWebViewAccessibleGetIndexInParent(AtkObject* atkObject)
{
    AtkObject* atkParent = atk_object_get_parent(atkObject);
    if (!atkParent)
        return -1;

    guint count = atk_object_get_n_accessible_children(atkParent);
    for (guint i = 0; i < count; ++i) {
        AtkObject* child = atk_object_ref_accessible_child(atkParent, i);
        bool childIsObject = child == atkObject;
        g_object_unref(child);
        if (childIsObject)
            return i;
    }

    return -1;
}

static void webkit_web_view_accessible_class_init(WebKitWebViewAccessibleClass* klass)
{
    // No need to implement get_n_children() and ref_child() here
    // since this is a subclass of AtkSocket and all the logic related
    // to those functions will be implemented by the ATK bridge.
    AtkObjectClass* atkObjectClass = ATK_OBJECT_CLASS(klass);
    atkObjectClass->initialize = webkitWebViewAccessibleInitialize;
    atkObjectClass->ref_state_set = webkitWebViewAccessibleRefStateSet;
    atkObjectClass->get_index_in_parent = webkitWebViewAccessibleGetIndexInParent;
}

WebKitWebViewAccessible* webkitWebViewAccessibleNew(gpointer webView)
{
    AtkObject* object = ATK_OBJECT(g_object_new(WEBKIT_TYPE_WEB_VIEW_ACCESSIBLE, nullptr));
    atk_object_initialize(object, webView);
    return WEBKIT_WEB_VIEW_ACCESSIBLE(object);
}

void webkitWebViewAccessibleSetWebView(WebKitWebViewAccessible* accessible, gpointer webView)
{
    g_return_if_fail(WEBKIT_IS_WEB_VIEW_ACCESSIBLE(accessible));

    if (accessible->priv->webView == webView)
        return;

    if (accessible->priv->webView && !webView)
        atk_object_notify_state_change(ATK_OBJECT(accessible), ATK_STATE_DEFUNCT, TRUE);

    bool didHaveWebView = accessible->priv->webView;
    accessible->priv->webView = webView;

    if (!didHaveWebView && webView)
        atk_object_notify_state_change(ATK_OBJECT(accessible), ATK_STATE_DEFUNCT, FALSE);
}

#endif // USE(ATK)
