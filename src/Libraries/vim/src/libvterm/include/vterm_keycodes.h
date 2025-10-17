/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

#ifndef __VTERM_INPUT_H__
#define __VTERM_INPUT_H__

typedef enum {
  VTERM_MOD_NONE  = 0x00,
  VTERM_MOD_SHIFT = 0x01,
  VTERM_MOD_ALT   = 0x02,
  VTERM_MOD_CTRL  = 0x04,

  VTERM_ALL_MODS_MASK = 0x07
} VTermModifier;

// The order here must match keycodes[] in src/keyboard.c!
typedef enum {
  VTERM_KEY_NONE,

  VTERM_KEY_ENTER,
  VTERM_KEY_TAB,
  VTERM_KEY_BACKSPACE,
  VTERM_KEY_ESCAPE,

  VTERM_KEY_UP,
  VTERM_KEY_DOWN,
  VTERM_KEY_LEFT,
  VTERM_KEY_RIGHT,

  VTERM_KEY_INS,
  VTERM_KEY_DEL,
  VTERM_KEY_HOME,
  VTERM_KEY_END,
  VTERM_KEY_PAGEUP,
  VTERM_KEY_PAGEDOWN,

  // F1 is VTERM_KEY_FUNCTION(1), F2 VTERM_KEY_FUNCTION(2), etc.
  VTERM_KEY_FUNCTION_0   = 256,
  VTERM_KEY_FUNCTION_MAX = VTERM_KEY_FUNCTION_0 + 255,

  // keypad keys
  VTERM_KEY_KP_0,
  VTERM_KEY_KP_1,
  VTERM_KEY_KP_2,
  VTERM_KEY_KP_3,
  VTERM_KEY_KP_4,
  VTERM_KEY_KP_5,
  VTERM_KEY_KP_6,
  VTERM_KEY_KP_7,
  VTERM_KEY_KP_8,
  VTERM_KEY_KP_9,
  VTERM_KEY_KP_MULT,
  VTERM_KEY_KP_PLUS,
  VTERM_KEY_KP_COMMA,
  VTERM_KEY_KP_MINUS,
  VTERM_KEY_KP_PERIOD,
  VTERM_KEY_KP_DIVIDE,
  VTERM_KEY_KP_ENTER,
  VTERM_KEY_KP_EQUAL,

  VTERM_KEY_MAX, // Must be last
  VTERM_N_KEYS = VTERM_KEY_MAX
} VTermKey;

#define VTERM_KEY_FUNCTION(n) (VTERM_KEY_FUNCTION_0+(n))

#endif
