/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
/****************************************************************************
 *   Author:  Nicolas Boulenguez, 2011                                      *
 ****************************************************************************/

#include "c_threaded_variables.h"

#define WRAP(type, name)        \
  type                          \
  name ## _as_function ()       \
  {                             \
    return name;                \
  }
/* *INDENT-OFF* */
WRAP(WINDOW *, stdscr)
WRAP(WINDOW *, curscr)

WRAP(int, LINES)
WRAP(int, COLS)
WRAP(int, TABSIZE)
WRAP(int, COLORS)
WRAP(int, COLOR_PAIRS)

chtype
acs_map_as_function(char inx)
{
  return acs_map[(unsigned char) inx];
}
/* *INDENT-ON* */
