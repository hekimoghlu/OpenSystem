/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
/* Put strings representing decimal floating-point numbers
 * into canonical form: always have a decimal point or
 * exponent field; if using an exponent field, have the
 * number before it start with a digit and decimal point
 * (if the number has more than one digit); only have an
 * exponent field if it saves space.
 *
 * Arrange that the return value, rv, satisfies rv[0] == '-' || rv[-1] == '-' .
 */

#include "defs.h"

 char *
#ifdef KR_headers
cds(s, z0)
	char *s;
	char *z0;
#else
cds(char *s, char *z0)
#endif
{
	int ea, esign, et, i, k, nd = 0, sign = 0, tz;
	char c, *z;
	char ebuf[24];
	long ex = 0;
	static char etype[Table_size], *db;
	static int dblen = 64;

	if (!db) {
		etype['E'] = 1;
		etype['e'] = 1;
		etype['D'] = 1;
		etype['d'] = 1;
		etype['+'] = 2;
		etype['-'] = 3;
		db = Alloc(dblen);
		}

	while((c = *s++) == '0');
	if (c == '-')
		{ sign = 1; c = *s++; }
	else if (c == '+')
		c = *s++;
	k = strlen(s) + 2;
	if (k >= dblen) {
		do dblen <<= 1;
			while(k >= dblen);
		free(db);
		db = Alloc(dblen);
		}
	if (etype[(unsigned char)c] >= 2)
		while(c == '0') c = *s++;
	tz = 0;
	while(c >= '0' && c <= '9') {
		if (c == '0')
			tz++;
		else {
			if (nd)
				for(; tz; --tz)
					db[nd++] = '0';
			else
				tz = 0;
			db[nd++] = c;
			}
		c = *s++;
		}
	ea = -tz;
	if (c == '.') {
		while((c = *s++) >= '0' && c <= '9') {
			if (c == '0')
				tz++;
			else {
				if (tz) {
					ea += tz;
					if (nd)
						for(; tz; --tz)
							db[nd++] = '0';
					else
						tz = 0;
					}
				db[nd++] = c;
				ea++;
				}
			}
		}
	if (et = etype[(unsigned char)c]) {
		esign = et == 3;
		c = *s++;
		if (et == 1) {
			if(etype[(unsigned char)c] > 1) {
				if (c == '-')
					esign = 1;
				c = *s++;
				}
			}
		while(c >= '0' && c <= '9') {
			ex = 10*ex + (c - '0');
			c = *s++;
			}
		if (esign)
			ex = -ex;
		}
	switch(c) {
		case 0:
			break;
#ifndef VAX
		case 'i':
		case 'I':
			Fatal("Overflow evaluating constant expression.");
		case 'n':
		case 'N':
			Fatal("Constant expression yields NaN.");
#endif
		default:
			Fatal("unexpected character in cds.");
		}
	ex -= ea;
	if (!nd) {
		if (!z0)
			z0 = mem(4,0);
		strcpy(z0, "-0.");
		/* sign = 0; */ /* 20010820: preserve sign of 0. */
		}
	else if (ex > 2 || ex + nd < -2) {
		sprintf(ebuf, "%ld", ex + nd - 1);
		k = strlen(ebuf) + nd + 3;
		if (nd > 1)
			k++;
		if (!z0)
			z0 = mem(k,0);
		z = z0;
		*z++ = '-';
		*z++ = *db;
		if (nd > 1) {
			*z++ = '.';
			for(k = 1; k < nd; k++)
				*z++ = db[k];
			}
		*z++ = 'e';
		strcpy(z, ebuf);
		}
	else {
		k = (int)(ex + nd);
		i = nd + 3;
		if (k < 0)
			i -= k;
		else if (ex > 0)
			i += (int)ex;
		if (!z0)
			z0 = mem(i,0);
		z = z0;
		*z++ = '-';
		if (ex >= 0) {
			for(k = 0; k < nd; k++)
				*z++ = db[k];
			while(--ex >= 0)
				*z++ = '0';
			*z++ = '.';
			}
		else {
			for(i = 0; i < k;)
				*z++ = db[i++];
			*z++ = '.';
			while(++k <= 0)
				*z++ = '0';
			while(i < nd)
				*z++ = db[i++];
			}
		*z = 0;
		}
	return sign ? z0 : z0+1;
	}
