/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1995                    *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 *     and: Juergen Pfeifer                         1996-1999,2008          *
 *     and: Thomas E. Dickey                                                *
 ****************************************************************************/

/* panel.c -- implementation of panels library, some core routines */
#include "panel.priv.h"

MODULE_ID("$Id: panel.c,v 1.26 2012/02/23 10:02:15 tom Exp $")

/*+-------------------------------------------------------------------------
	_nc_retrace_panel (pan)
--------------------------------------------------------------------------*/
#ifdef TRACE
NCURSES_EXPORT(PANEL *)
_nc_retrace_panel(PANEL * pan)
{
  T((T_RETURN("%p"), (void *)pan));
  return pan;
}
#endif

/*+-------------------------------------------------------------------------
	_nc_my_visbuf(ptr)
--------------------------------------------------------------------------*/
#ifdef TRACE
#ifndef TRACE_TXT
NCURSES_EXPORT(const char *)
_nc_my_visbuf(const void *ptr)
{
  char temp[32];

  if (ptr != 0)
    _nc_SPRINTF(temp, _nc_SLIMIT(sizeof(temp)) "ptr:%p", ptr);
  else
    _nc_STRCPY(temp, "<null>", sizeof(temp));
  return _nc_visbuf(temp);
}
#endif
#endif

/*+-------------------------------------------------------------------------
	dPanel(text,pan)
--------------------------------------------------------------------------*/
#ifdef TRACE
NCURSES_EXPORT(void)
_nc_dPanel(const char *text, const PANEL * pan)
{
  _tracef("%s id=%s b=%s a=%s y=%d x=%d",
	  text, USER_PTR(pan->user),
	  (pan->below) ? USER_PTR(pan->below->user) : "--",
	  (pan->above) ? USER_PTR(pan->above->user) : "--",
	  PSTARTY(pan), PSTARTX(pan));
}
#endif

/*+-------------------------------------------------------------------------
	dStack(fmt,num,pan)
--------------------------------------------------------------------------*/
#ifdef TRACE
NCURSES_EXPORT(void)
_nc_dStack(const char *fmt, int num, const PANEL * pan)
{
  char s80[80];

  GetPanelHook(pan);

  _nc_SPRINTF(s80, _nc_SLIMIT(sizeof(s80)) fmt, num, pan);
  _tracef("%s b=%s t=%s", s80,
	  (_nc_bottom_panel) ? USER_PTR(_nc_bottom_panel->user) : "--",
	  (_nc_top_panel) ? USER_PTR(_nc_top_panel->user) : "--");
  if (pan)
    _tracef("pan id=%s", USER_PTR(pan->user));
  pan = _nc_bottom_panel;
  while (pan)
    {
      dPanel("stk", pan);
      pan = pan->above;
    }
}
#endif

/*+-------------------------------------------------------------------------
	Wnoutrefresh(pan) - debugging hook for wnoutrefresh
--------------------------------------------------------------------------*/
#ifdef TRACE
NCURSES_EXPORT(void)
_nc_Wnoutrefresh(const PANEL * pan)
{
  dPanel("wnoutrefresh", pan);
  wnoutrefresh(pan->win);
}
#endif

/*+-------------------------------------------------------------------------
	Touchpan(pan)
--------------------------------------------------------------------------*/
#ifdef TRACE
NCURSES_EXPORT(void)
_nc_Touchpan(const PANEL * pan)
{
  dPanel("Touchpan", pan);
  touchwin(pan->win);
}
#endif

/*+-------------------------------------------------------------------------
	Touchline(pan,start,count)
--------------------------------------------------------------------------*/
#ifdef TRACE
NCURSES_EXPORT(void)
_nc_Touchline(const PANEL * pan, int start, int count)
{
  char s80[80];

  _nc_SPRINTF(s80, _nc_SLIMIT(sizeof(s80)) "Touchline s=%d c=%d", start, count);
  dPanel(s80, pan);
  touchline(pan->win, start, count);
}
#endif

#ifndef TRACE
#  ifndef __GNUC__
     /* Some C compilers need something defined in a source file */
extern void _nc_dummy_panel(void);
void
_nc_dummy_panel(void)
{
}
#  endif
#endif
