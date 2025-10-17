/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
#ifdef HAVE_BRAILLE

extern void StartBraille __P((void));

struct braille_display
{
  struct display *bd_dpy;	/* display we are connected to */
  int   bd_start_braille;	/* screenrc var to request to turn braille on */
  int   bd_using_braille;	/* all is fine, use braille */
  struct event bd_readev;
  struct event bd_writeev;
  struct event bd_selectev;
  int   bd_fd;		/* file descriptor */
  int   bd_obuflen;	/* current number of charactors in output buffer */
  char  bd_obuf[IOSIZE]; 
  int   bd_info;	/* default, no info, 0,1,2,3 */
  int   bd_ncrc;	/* default 1, numbers of cells on the right side of real cursor, 1...bd_width */
  int   bd_skip;	/* default off, on/off */
  int   bd_link;	/* default, linked, on/off */
  int   bd_width;	/* length of braille display to use, <=bd_ncells */
  int   bd_scroll;	/* default on, scroll on/off */
  char *bd_braille_table;	/* braille code */

  int   bd_bell;	/* bell used for navigation on/off */
  int   bd_ncells;	/* real number of cells on braille display */
  int   bd_eightdot;	/* eightdot on/off */
  int   bd_baud;	/* communication baudrate between port and braille display */
  char *bd_port;	/* serial port to use */
  char *bd_type; 	/* kind of braille display */
  double bd_version;	/* rom version of braille display */
  char  bd_btable[256];	/* braille translation table */

  /* functions which communicate with braille displays */
  int (*write_line_braille) __P((char [],int, int));
  void (*buttonpress) __P((void));
  int (*bd_response_test) __P((void));

  int   bd_refreshing;	/* are we doing a refresh? */
  char  bd_line[40+1];	/* bd_ncells chars displayed on braille */
  int   bd_cursorpos;	/* cursor position on braille */
  char  bd_oline[40+1];	/* bd_ncells chars displayed on braille */
  int   bd_sx, bd_sy;	/* screen cursor pos */
  int   bd_moved;	/* used braille move keys */

  int   bd_searching;	/* are we seaching (bd_skip is on) */
  int   bd_searchmax;	/* search: max x */
  int   bd_searchmin;	/* search: min x */
  int   bd_searchstart;
  int   bd_searchend;
};

extern struct braille_display bd;

#define BD_FORE bd.bd_dpy->d_fore

#endif
