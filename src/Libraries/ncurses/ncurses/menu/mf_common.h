/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
 *   Author:  Juergen Pfeifer, 1995,1997                                    *
 ****************************************************************************/

/* $Id: mf_common.h,v 0.24 2012/06/10 00:06:54 tom Exp $ */

/* Common internal header for menu and form library */

#ifndef MF_COMMON_H_incl
#define MF_COMMON_H_incl 1

#include <ncurses_cfg.h>
#include <curses.h>

#include <stdlib.h>
#include <sys/types.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#if DECL_ERRNO
extern int errno;
#endif

/* in case of debug version we ignore the suppression of assertions */
#ifdef TRACE
#  ifdef NDEBUG
#    undef NDEBUG
#  endif
#endif

#include <nc_alloc.h>

#if USE_RCS_IDS
#define MODULE_ID(id) static const char Ident[] = id;
#else
#define MODULE_ID(id)		/*nothing */
#endif

/* Maximum regular 8-bit character code */
#define MAX_REGULAR_CHARACTER (0xff)

#define SET_ERROR(code) (errno=(code))
#define GET_ERROR()     (errno)

#ifdef TRACE
#define RETURN(code)    returnCode( SET_ERROR(code) )
#else
#define RETURN(code)    return( SET_ERROR(code) )
#endif

/* The few common values in the status fields for menus and forms */
#define _POSTED         (0x01U)	/* menu or form is posted                  */
#define _IN_DRIVER      (0x02U)	/* menu or form is processing hook routine */

#define SetStatus(target,mask) (target)->status |= (unsigned short) (mask)
#define ClrStatus(target,mask) (target)->status = (unsigned short) (target->status & (~mask))

/* Call object hook */
#define Call_Hook( object, handler ) \
   if ( (object) != 0 && ((object)->handler) != (void *) 0 )\
   {\
	SetStatus(object, _IN_DRIVER);\
	(object)->handler(object);\
	ClrStatus(object, _IN_DRIVER);\
   }

#endif /* MF_COMMON_H_incl */
