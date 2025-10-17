/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
 * Author: Juergen Pfeifer, 2008-on                                         * 
 *                                                                          *
 ****************************************************************************/

/* $Id: ncurses_mingw.h,v 1.3 2014/05/03 19:40:19 juergen Exp $ */

/*
 * This is a placeholder up to now and describes what needs to be implemented
 * to support I/O to external terminals with ncurses on the Windows OS.
 */

#if __MINGW32__
#ifndef _NC_MINGWH
#define _NC_MINGWH

#define USE_CONSOLE_DRIVER 1

#undef  TERMIOS
#define TERMIOS 1

typedef unsigned char cc_t;
typedef unsigned int  tcflag_t;
typedef unsigned int  speed_t;
typedef unsigned short otcflag_t;
typedef unsigned char ospeed_t;

#define NCCS 18
struct termios
{
  tcflag_t	c_iflag;
  tcflag_t	c_oflag;
  tcflag_t	c_cflag;
  tcflag_t	c_lflag;
  char		c_line;
  cc_t		c_cc[NCCS];
  speed_t	c_ispeed;
  speed_t	c_ospeed;
};

extern NCURSES_EXPORT(int)  _nc_mingw_tcsetattr(
    int fd, 
    int optional_actions, 
    const struct termios* arg);
extern NCURSES_EXPORT(int)  _nc_mingw_tcgetattr(
    int fd, 
    struct termios* arg);
extern NCURSES_EXPORT(int)  _nc_mingw_tcflush(
    int fd, 
    int queue);
extern NCURSES_EXPORT(void) _nc_set_term_driver(void* term);

#endif /* _NC_MINGWH */
#endif /* __MINGW32__ */
