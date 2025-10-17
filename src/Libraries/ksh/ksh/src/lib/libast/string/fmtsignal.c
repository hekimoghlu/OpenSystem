/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Bell Laboratories
 *
 * if sig>=0 then return signal text for signal sig
 * otherwise return signal name for signal -sig
 */

#include <ast.h>
#include <sig.h>

char*
fmtsignal(register int sig)
{
	char*	buf;
	int	z;

	if (sig >= 0)
	{
		if (sig <= sig_info.sigmax)
			buf = sig_info.text[sig];
		else
		{
			buf = fmtbuf(z = 20);
			sfsprintf(buf, z, "Signal %d", sig);
		}
	}
	else
	{
		sig = -sig;
		if (sig <= sig_info.sigmax)
			buf = sig_info.name[sig];
		else
		{
			buf = fmtbuf(z = 20);
			sfsprintf(buf, z, "%d", sig);
		}
	}
	return buf;
}
