/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
// GtkEmojiChooser is private in GTK 3, so this goes through some hoops to
// obtain its type code via g_type_from_name(). The main issue is that the
// type needs to have been previously registered, which in practice means
// triggering a call to gtk_emoji_chooser_get_type(), also a hidden private
// function.
//
// One point in the public API where an emoji chooser can be explicitly
// triggered is the GtkEntry.insert_emoji vfunc: either at that point the
// GtkEmojiChooser used by the entry has been already created, or it will
// be created by calling it. Object instantiation needs the type to be
// registered, so after making sure at least one instance is created it
// is safe to use g_type_from_name().

#include "config.h"
#include "WebKitEmojiChooser.h"

#if GTK_CHECK_VERSION(3, 24, 0) && !USE(GTK4)

#include <wtf/RunLoop.h>
#include <wtf/glib/GRefPtr.h>

GtkWidget* webkitEmojiChooserNew()
{
    static GType chooserType = ([] {
        GRefPtr<GtkWidget> entry = gtk_entry_new();
        gtk_entry_set_input_hints(GTK_ENTRY(entry.get()), GTK_INPUT_HINT_EMOJI);

        GtkEntryClass* entryClass = GTK_ENTRY_GET_CLASS(entry.get());
        entryClass->insert_emoji(GTK_ENTRY(entry.get()));

        // The emoji data is fetched in a delayed manner, so a direct call
        // to g_object_unref(entry) results in critical warning due to an
        // attempt to unref a yet unpopulated value. Arguably, GTK should do
        // a null-check, but this code needs to work even for older versions
        // of GTK which may be unpatched. Therefore, destroy the dummy entry
        // in a low priority idle source, after the source that fills the data
        // has been already dispatched.
        GRefPtr<GSource> source = adoptGRef(g_idle_source_new());
        g_source_set_callback(source.get(), [](void* data) {
            g_object_unref(data);
            return G_SOURCE_REMOVE;
        }, entry.leakRef(), nullptr);
        g_source_set_priority(source.get(), G_PRIORITY_LOW);
        g_source_attach(source.get(), RunLoop::main().mainContext());

        return g_type_from_name("GtkEmojiChooser");
    })();

    return GTK_WIDGET(g_object_new(chooserType, nullptr));
}

#endif // GTK_CHECK_VERSION(3, 24, 0) && !USE(GTK4)
