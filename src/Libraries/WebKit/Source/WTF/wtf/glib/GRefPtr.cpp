/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#include <wtf/glib/GRefPtr.h>

#if USE(GLIB)

#include <gio/gio.h>

namespace WTF {

template <> GHashTable* refGPtr(GHashTable* ptr)
{
    if (ptr)
        g_hash_table_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GHashTable* ptr)
{
    if (ptr)
        g_hash_table_unref(ptr);
}

template <> GMainContext* refGPtr(GMainContext* ptr)
{
    if (ptr)
        g_main_context_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GMainContext* ptr)
{
    if (ptr)
        g_main_context_unref(ptr);
}

template <> GMainLoop* refGPtr(GMainLoop* ptr)
{
    if (ptr)
        g_main_loop_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GMainLoop* ptr)
{
    if (ptr)
        g_main_loop_unref(ptr);
}

template <> GBytes* refGPtr(GBytes* ptr)
{
    if (ptr)
        g_bytes_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GBytes* ptr)
{
    if (ptr)
        g_bytes_unref(ptr);
}

template <> GVariant* refGPtr(GVariant* ptr)
{
    if (ptr)
        g_variant_ref_sink(ptr);
    return ptr;
}

template <> void derefGPtr(GVariant* ptr)
{
    if (ptr)
        g_variant_unref(ptr);
}

template <> GVariantBuilder* refGPtr(GVariantBuilder* ptr)
{
    if (ptr)
        g_variant_builder_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GVariantBuilder* ptr)
{
    if (ptr)
        g_variant_builder_unref(ptr);
}

template <> GSource* refGPtr(GSource* ptr)
{
    if (ptr)
        g_source_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GSource* ptr)
{
    if (ptr)
        g_source_unref(ptr);
}

template <> GPtrArray* refGPtr(GPtrArray* ptr)
{
    if (ptr)
        g_ptr_array_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GPtrArray* ptr)
{
    if (ptr)
        g_ptr_array_unref(ptr);
}

template <> GByteArray* refGPtr(GByteArray* ptr)
{
    if (ptr)
        g_byte_array_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GByteArray* ptr)
{
    if (ptr)
        g_byte_array_unref(ptr);
}

template <> GClosure* refGPtr(GClosure* ptr)
{
    if (ptr)
        g_closure_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GClosure* ptr)
{
    if (ptr)
        g_closure_unref(ptr);
}

template <> GRegex* refGPtr(GRegex* ptr)
{
    if (ptr)
        g_regex_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GRegex* ptr)
{
    if (ptr)
        g_regex_unref(ptr);
}

template <> GMappedFile* refGPtr(GMappedFile* ptr)
{
    if (ptr)
        g_mapped_file_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GMappedFile* ptr)
{
    if (ptr)
        g_mapped_file_unref(ptr);
}

template <> GDateTime* refGPtr(GDateTime* ptr)
{
    if (ptr)
        g_date_time_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GDateTime* ptr)
{
    if (ptr)
        g_date_time_unref(ptr);
}

template <> GDBusNodeInfo* refGPtr(GDBusNodeInfo* ptr)
{
    if (ptr)
        g_dbus_node_info_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GDBusNodeInfo* ptr)
{
    if (ptr)
        g_dbus_node_info_unref(ptr);
}

template <> GUri* refGPtr(GUri* ptr)
{
    if (ptr)
        g_uri_ref(ptr);
    return ptr;
}

template <> void derefGPtr(GUri* ptr)
{
    if (ptr)
        g_uri_unref(ptr);
}

template <>
GArray* refGPtr(GArray* ptr)
{
    if (ptr)
        g_array_ref(ptr);

    return ptr;
}

template <>
void derefGPtr(GArray* ptr)
{
    if (ptr)
        g_array_unref(ptr);
}

template <>
GResource* refGPtr(GResource* ptr)
{
    if (ptr)
        g_resource_ref(ptr);
    return ptr;
}

template <>
void derefGPtr(GResource* ptr)
{
    if (ptr)
        g_resource_unref(ptr);
}

} // namespace WTF

#endif // USE(GLIB)
