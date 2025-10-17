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
/****************************************************************************
 *  Author: Thomas E. Dickey <dickey@clark.net> 1999                        *
 ****************************************************************************/
/*
 *	trace_xnames.c - Tracing/Debugging buffers (TERMTYPE extended names)
 */

#include <curses.priv.h>

MODULE_ID("$Id: trace_xnames.c,v 1.6 2010/01/23 17:59:27 tom Exp $")

NCURSES_EXPORT(void)
_nc_trace_xnames(TERMTYPE *tp GCC_UNUSED)
{
#ifdef TRACE
#if NCURSES_XNAMES
    int limit = tp->ext_Booleans + tp->ext_Numbers + tp->ext_Strings;
    int n, m;
    if (limit) {
	int begin_num = tp->ext_Booleans;
	int begin_str = tp->ext_Booleans + tp->ext_Numbers;

	_tracef("extended names (%s) %d = %d+%d+%d of %d+%d+%d",
		tp->term_names,
		limit,
		tp->ext_Booleans, tp->ext_Numbers, tp->ext_Strings,
		tp->num_Booleans, tp->num_Numbers, tp->num_Strings);
	for (n = 0; n < limit; n++) {
	    if ((m = n - begin_str) >= 0) {
		_tracef("[%d] %s = %s", n,
			tp->ext_Names[n],
			_nc_visbuf(tp->Strings[tp->num_Strings + m - tp->ext_Strings]));
	    } else if ((m = n - begin_num) >= 0) {
		_tracef("[%d] %s = %d (num)", n,
			tp->ext_Names[n],
			tp->Numbers[tp->num_Numbers + m - tp->ext_Numbers]);
	    } else {
		_tracef("[%d] %s = %d (bool)", n,
			tp->ext_Names[n],
			tp->Booleans[tp->num_Booleans + n - tp->ext_Booleans]);
	    }
	}
    }
#endif
#endif
}
