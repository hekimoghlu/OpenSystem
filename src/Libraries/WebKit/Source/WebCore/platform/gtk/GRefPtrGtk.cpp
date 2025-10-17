/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#include "GRefPtrGtk.h"

#include <glib.h>
#include <gtk/gtk.h>

#if USE(LIBSECRET)
#define SECRET_WITH_UNSTABLE 1
#define SECRET_API_SUBJECT_TO_CHANGE 1
#include <libsecret/secret.h>
#endif

namespace WTF {

#if !USE(GTK4)
template <> GtkTargetList* refGPtr(GtkTargetList* ptr)
{
    if (ptr)
        gtk_target_list_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GtkTargetList* ptr)
{
    if (ptr)
        gtk_target_list_unref(ptr);
}
#endif

#if USE(LIBSECRET)
template <> SecretValue* refGPtr(SecretValue* ptr)
{
    if (ptr)
        secret_value_ref(ptr);
    return ptr;
}

template <> void derefGPtr(SecretValue* ptr)
{
    if (ptr)
        secret_value_unref(ptr);
}
#endif

#if !USE(GTK4)
template <> GtkWidgetPath* refGPtr(GtkWidgetPath* ptr)
{
    if (ptr)
        gtk_widget_path_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GtkWidgetPath* ptr)
{
    if (ptr)
        gtk_widget_path_unref(ptr);
}
#endif

#if USE(GTK4)
template <> GskRenderNode* refGPtr(GskRenderNode* ptr)
{
    if (ptr)
        gsk_render_node_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GskRenderNode* ptr)
{
    if (ptr)
        gsk_render_node_unref(ptr);
}

template <> GdkEvent* refGPtr(GdkEvent* ptr)
{
    if (ptr)
        gdk_event_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GdkEvent* ptr)
{
    if (ptr)
        gdk_event_unref(ptr);
}
#endif

}
