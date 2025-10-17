/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
/* Copyright (C) 1997 Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2, or (at your option) any later
   version.

   Bash is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with Bash; see the file COPYING.  If not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA */

#include <config.h>

#include <bashtypes.h>
#include <sys/param.h>

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#if defined (HAVE_LIMITS_H)
#  include <limits.h>
#endif

#if !defined (HAVE_SYSCONF) || !defined (_SC_CLK_TCK)
#  if !defined (CLK_TCK)
#    if defined (HZ)
#      define CLK_TCK HZ
#    else
#      define CLK_TCK 60
#    endif
#  endif /* !CLK_TCK */
#endif /* !HAVE_SYSCONF && !_SC_CLK_TCK */

long
get_clk_tck ()
{
  static long retval = 0;

  if (retval != 0)
    return (retval);

#if defined (HAVE_SYSCONF) && defined (_SC_CLK_TCK)
  retval = sysconf (_SC_CLK_TCK);
#else /* !SYSCONF || !_SC_CLK_TCK */
  retval = CLK_TCK;
#endif /* !SYSCONF || !_SC_CLK_TCK */

  return (retval);
}
