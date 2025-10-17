/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
/* $Id: c_threaded_variables.h,v 1.3 2015/08/06 23:09:47 tom Exp $ */

#ifndef __C_THREADED_VARIABLES_H
#define __C_THREADED_VARIABLES_H

#include <ncurses_cfg.h>

#if HAVE_INTTYPES_H
# include <inttypes.h>
#else
# if HAVE_STDINT_H
#  include <stdint.h>
# endif
#endif

#include <curses.h>

extern WINDOW *stdscr_as_function(void);
extern WINDOW *curscr_as_function(void);

extern int LINES_as_function(void);
extern int LINES_as_function(void);
extern int COLS_as_function(void);
extern int TABSIZE_as_function(void);
extern int COLORS_as_function(void);
extern int COLOR_PAIRS_as_function(void);

extern chtype acs_map_as_function(char /* index */ );

#endif /* __C_THREADED_VARIABLES_H */
