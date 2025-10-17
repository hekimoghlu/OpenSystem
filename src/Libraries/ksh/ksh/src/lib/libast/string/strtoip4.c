/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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

#include <ast.h>
#include <ctype.h>

/*
 * convert string to 4 byte local byte order ip address
 * with optional prefix bits
 * pointer to first unused char placed in *e, even on error
 * return 0:ok <0:error
 *
 * valid addresses match the egrep RE:
 *
 *	[0-9]{1,3}(\.[0-9]{1,3})*|0[xX][0-9a-fA-Z]+
 *
 * valid bits/masks match the egrep RE:
 *
 *	(/([0-9]+|[0-9]{1,3}(\.[0-9]{1,3})*))?
 *
 * if pbits!=0 and no bits/mask specified then trailing 0's in addr
 * are used to compute the mask
 */

int
strtoip4(register const char* s, char** e, uint32_t* paddr, unsigned char* pbits)
{
	register int		c;
	register unsigned int	n;
	register uint32_t	addr;
	register int		part;
	register unsigned char	bits;
	uint32_t		z;
	int			old;
	int			r;
	const char*		b;

	r = -1;
	while (isspace(*s))
		s++;
	b = s;
	addr = 0;
	bits = 0;
	part = 0;
	do
	{
		n = 0;
		while ((c = *s++) >= '0' && c <= '9')
			n = n * 10 + (c - '0');
		if ((c == 'x' || c == 'X') && !part)
		{
			addr = n;
			for (;;)
			{
				if ((c = *s++) >= '0' && c <= '9')
					c -= '0';
				else if (c >= 'a' && c <= 'f')
					c -= 'a' - 10;
				else if (c >= 'A' && c <= 'F')
					c -= 'F' - 10;
				else
					break;
				addr = addr * 16 + c;
			}
			part = 4;
			break;
		}
		if (n > 0xff)
			goto done;
		addr = (addr << 8) | n;
		part++;
	} while (c == '.');
	if ((s - b) == 1 && c != '/' || part > 4)
		goto done;
	if (old = part < 4)
		while (part++ < 4)
			addr <<= 8;
	if (pbits)
	{
		if (c == '/')
		{
			part = 0;
			z = 0;
			for (;;)
			{
				n = 0;
				while ((c = *s++) >= '0' && c <= '9')
					n = n * 10 + (c - '0');
				z = (z << 8) | n;
				part++;
				if (c != '.')
					break;
				old = 1;
			}
			if (part > 4)
				goto done;
			if (z <= 32 && (!old || part < 2))
				bits = z;
			else if (z)
			{
				if (part == 4 && (z & 0x8000001) == 1)
					z = ~z;
				while (!(z & 1))
					z >>= 1;
				while (z & 1)
				{
					z >>= 1;
					bits++;
				}
			}
		}
		else if ((z = (addr >> 24)) < 128)
			bits = 8;
		else if (z < 192)
			bits = 16;
		else
			bits = 24;
		if (*pbits = bits)
			addr &= ~((((uint32_t)1)<<(32-bits))-1);
		else
			addr = 0;
	}
	if (paddr)
		*paddr = addr;
	r = 0;
 done:
	if (e)
		*e = (char*)(s - 1);
	return r;
}
