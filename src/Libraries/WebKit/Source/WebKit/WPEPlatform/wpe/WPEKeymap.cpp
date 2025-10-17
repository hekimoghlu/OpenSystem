/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include "WPEKeymap.h"

#include <wtf/glib/WTFGType.h>

/**
 * WPEKeymap:
 *
 */
struct _WPEKeymapPrivate {
};

WEBKIT_DEFINE_ABSTRACT_TYPE(WPEKeymap, wpe_keymap, G_TYPE_OBJECT)

static void wpe_keymap_class_init(WPEKeymapClass*)
{
}

/**
 * wpe_keymap_get_entries_for_keycode:
 * @keymap: a #WPEKaymap
 * @keyval: a keyval
 * @entries: (out): return location for array of #WPEKeymapEntry
 * @n_entries: (out): return location for length of @entries
 *
 * Get the @keymap list of keycode/group/level combinations that will generate @keyval
 *
 * Returns: %TRUE if there were entries, or %FALSE otherwise
 */
gboolean wpe_keymap_get_entries_for_keyval(WPEKeymap* keymap, guint keyval, WPEKeymapEntry** entries, guint *entriesCount)
{
    g_return_val_if_fail(WPE_IS_KEYMAP(keymap), FALSE);
    g_return_val_if_fail(entries, FALSE);
    g_return_val_if_fail(entriesCount, FALSE);

    return WPE_KEYMAP_GET_CLASS(keymap)->get_entries_for_keyval(keymap, keyval, entries, entriesCount);
}

/**
 * wpe_keymap_translate_keyboard_state:
 * @keymap: a #WPEKaymap
 * @keycode: a hardware keycode
 * @modifiers: a #WPEModifiers
 * @group: active keyboard group
 * @keyval: (out) (optional): return location for keyval
 * @effective_group: (out) (optional): return location for effective group
 * @level: (out) (optional): return location for level
 * @consumed_modifiers: (out) (optional): return location for modifiers that were used to determine the group or level
 *
 * Translate @keycode, @modifiers and @group into a keyval, effective group and level.
 * Modifiers that affected the translation are returned in @consumed_modifiers.
 *
 * Returns: %TRUE if there was a keyval bound to keycode, modifiers and group, or %FALSE otherwise
 */
gboolean wpe_keymap_translate_keyboard_state(WPEKeymap* keymap, guint keycode, WPEModifiers modifiers, int group, guint* keyval, int* effectiveGroup, int* level, WPEModifiers* consumedModifiers)
{
    g_return_val_if_fail(WPE_IS_KEYMAP(keymap), FALSE);

    return WPE_KEYMAP_GET_CLASS(keymap)->translate_keyboard_state(keymap, keycode, modifiers, group, keyval, effectiveGroup, level, consumedModifiers);
}

/**
 * wpe_keymap_get_modifiers:
 * @keymap: a #WPEKaymap
 *
 * Get the modifiers state of @keymap
 *
 * Returns: a #WPEModifiers
 */
WPEModifiers wpe_keymap_get_modifiers(WPEKeymap* keymap)
{
    g_return_val_if_fail(WPE_IS_KEYMAP(keymap), static_cast<WPEModifiers>(0));

    return WPE_KEYMAP_GET_CLASS(keymap)->get_modifiers(keymap);
}
