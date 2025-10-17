/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#ifndef WPEKeymap_h
#define WPEKeymap_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>
#include <wpe/WPEEvent.h>

G_BEGIN_DECLS

#define WPE_TYPE_KEYMAP (wpe_keymap_get_type())
WPE_DECLARE_DERIVABLE_TYPE (WPEKeymap, wpe_keymap, WPE, KEYMAP, GObject)

typedef struct _WPEKeymapEntry WPEKeymapEntry;

/**
 * WPEKeymapEntry:
 * @keycode: the hardware keycode. This is an identifying number for a physical key.
 * @group: indicates movement in a horizontal direction. Usually groups are used
 *   for two different languages. In group 0, a key might have two English
 *   characters, and in group 1 it might have two Hebrew characters. The Hebrew
 *   characters will be printed on the key next to the English characters.
 * @level: indicates which symbol on the key will be used, in a vertical direction.
 *   So on a standard US keyboard, the key with the number â€œ1â€ on it also has the
 *   exclamation point ("!") character on it. The level indicates whether to use
 *   the â€œ1â€ or the â€œ!â€ symbol. The letter keys are considered to have a lowercase
 *   letter at level 0, and an uppercase letter at level 1, though only the
 *   uppercase letter is printed.
 *
 * A WPEKeymapEntry is a map entry retrurned by wpe_keymap_get_entries_for_keyval().
 */
struct _WPEKeymapEntry
{
  guint keycode;
  int   group;
  int   level;
};

struct _WPEKeymapClass
{
    GObjectClass parent_class;

    gboolean     (* get_entries_for_keyval)   (WPEKeymap       *keymap,
                                               guint            keyval,
                                               WPEKeymapEntry **entries,
                                               guint           *n_entries);
    gboolean     (* translate_keyboard_state) (WPEKeymap       *keymap,
                                               guint            keycode,
                                               WPEModifiers     modifiers,
                                               int              group,
                                               guint           *keyval,
                                               int             *effective_group,
                                               int             *level,
                                               WPEModifiers    *consumed_modifiers);
    WPEModifiers (* get_modifiers)            (WPEKeymap       *keymap);

    gpointer padding[32];
};

WPE_API gboolean     wpe_keymap_get_entries_for_keyval   (WPEKeymap       *keymap,
                                                          guint            keyval,
                                                          WPEKeymapEntry **entries,
                                                          guint           *n_entries);
WPE_API gboolean     wpe_keymap_translate_keyboard_state (WPEKeymap       *keymap,
                                                          guint            keycode,
                                                          WPEModifiers     modifiers,
                                                          int              group,
                                                          guint           *keyval,
                                                          int             *effective_group,
                                                          int             *level,
                                                          WPEModifiers    *consumed_modifiers);
WPE_API WPEModifiers wpe_keymap_get_modifiers            (WPEKeymap       *keymap);

G_END_DECLS

#endif /* WPEKeymap_h */
