/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
 * return the record format string given a format descriptor
 */

#include <recfmt.h>
#include <ctype.h>

char*
fmtrec(Recfmt_t f, int fs)
{
	char*	b;
	char*	e;
	char*	s;
	long	n;
	char	del[2];

	b = s = fmtbuf(n = 32);
	e = b + n;
	switch (RECTYPE(f))
	{
	case REC_delimited:
		*s++ = 'd';
		if ((del[0] = REC_D_DELIMITER(f)) != '\n')
		{
			del[1] = 0;
			if (fs)
				sfsprintf(s, e - s, "0x%02x", *(unsigned char*)del);
			else
				sfsprintf(s, e - s, "%s", fmtquote(del, NiL, NiL, 1, 0));
		}
		else
			*s = 0;
		break;
	case REC_fixed:
		if (!fs)
			*s++ = 'f';
		sfsprintf(s, e - s, "%lu", REC_F_SIZE(f));
		break;
	case REC_variable:
		*s++ = 'v';
		if (n = REC_V_SIZE(f))
			s += sfsprintf(s, e - s, "%lu", n);
		if (REC_V_HEADER(f) != 4)
			s += sfsprintf(s, e - s, "h%u", REC_V_HEADER(f));
		if (REC_V_OFFSET(f) != 0)
			s += sfsprintf(s, e - s, "o%u", REC_V_OFFSET(f));
		if (REC_V_LENGTH(f) != 2)
			s += sfsprintf(s, e - s, "z%u", REC_V_LENGTH(f));
		if (REC_V_LITTLE(f) != 0)
			*s++ = 'l';
		if (REC_V_INCLUSIVE(f) == 0)
			*s++ = 'n';
		*s = 0;
		break;
	case REC_method:
		*s++ = 'm';
		switch (n = REC_M_INDEX(f))
		{
		case REC_M_data:
			sfsprintf(s, e - s, "data");
			break;
		case REC_M_path:
			sfsprintf(s, e - s, "path");
			break;
		default:
			sfsprintf(s, e - s, "%lu", n);
			break;
		}
		break;
	case REC_none:
		*s++ = 'n';
		*s = 0;
		break;
	default:
		sfsprintf(s, e - s, "u%u.0x%07x", RECTYPE(f), REC_U_ATTRIBUTES(f));
		break;
	}
	return b;
}
