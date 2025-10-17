/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#ifndef WPEKeymapXKB_h
#define WPEKeymapXKB_h

#if !defined(__WPE_PLATFORM_H_INSIDE__) && !defined(BUILDING_WEBKIT)
#error "Only <wpe/wpe-platform.h> can be included directly."
#endif

#include <glib-object.h>
#include <wpe/WPEDefines.h>
#include <wpe/WPEKeymap.h>
#include <xkbcommon/xkbcommon.h>

G_BEGIN_DECLS

#define WPE_TYPE_KEYMAP_XKB (wpe_keymap_xkb_get_type())
WPE_API G_DECLARE_FINAL_TYPE (WPEKeymapXKB, wpe_keymap_xkb, WPE, KEYMAP_XKB, WPEKeymap)

WPE_API WPEKeymap         *wpe_keymap_xkb_new            (void);
WPE_API void               wpe_keymap_xkb_update         (WPEKeymapXKB *keymap,
                                                          guint         format,
                                                          int           fd,
                                                          guint         size);
WPE_API struct xkb_keymap *wpe_keymap_xkb_get_xkb_keymap (WPEKeymapXKB *keymap);
WPE_API struct xkb_state  *wpe_keymap_xkb_get_xkb_state  (WPEKeymapXKB *keymap);

G_END_DECLS

#endif /* WPEKeymapXKB_h */
