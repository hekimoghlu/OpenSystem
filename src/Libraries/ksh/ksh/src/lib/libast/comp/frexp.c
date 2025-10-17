/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
 * frexp/ldexp implementation
 */

#include <ast.h>

#include "FEATURE/float"

#if _lib_frexp && _lib_ldexp

NoN(frexp)

#else

#if defined(_ast_dbl_exp_index) && defined(_ast_dbl_exp_shift)

#define INIT()		_ast_dbl_exp_t _pow_
#define pow2(i)		(_pow_.f=1,_pow_.e[_ast_dbl_exp_index]+=((i)<<_ast_dbl_exp_shift),_pow_.f)

#else

static double		pow2tab[DBL_MAX_EXP + 1];

static int
init(void)
{
	register int	x;
	double		g;

	g = 1;
	for (x = 0; x < elementsof(pow2tab); x++)
	{
		pow2tab[x] = g;
		g *= 2;
	}
	return 0;
}

#define INIT()		(pow2tab[0]?0:init())

#define pow2(i)		pow2tab[i]

#endif

#if !_lib_frexp

extern double
frexp(double f, int* p)
{
	register int	k;
	register int	x;
	double		g;

	INIT();

	/*
	 * normalize
	 */

	x = k = DBL_MAX_EXP / 2;
	if (f < 1)
	{
		g = 1.0L / f;
		for (;;)
		{
			k = (k + 1) / 2;
			if (g < pow2(x))
				x -= k;
			else if (k == 1 && g < pow2(x+1))
				break;
			else
				x += k;
		}
		if (g == pow2(x))
			x--;
		x = -x;
	}
	else if (f > 1)
	{
		for (;;)
		{
			k = (k + 1) / 2;
			if (f > pow2(x))
				x += k;
			else if (k == 1 && f > pow2(x-1))
				break;
			else
				x -= k;
		}
		if (f == pow2(x))
			x++;
	}
	else
		x = 1;
	*p = x;

	/*
	 * shift
	 */

	x = -x;
	if (x < 0)
		f /= pow2(-x);
	else if (x < DBL_MAX_EXP)
		f *= pow2(x);
	else
		f = (f * pow2(DBL_MAX_EXP - 1)) * pow2(x - (DBL_MAX_EXP - 1));
	return f;
}

#endif

#if !_lib_ldexp

extern double
ldexp(double f, register int x)
{
	INIT();
	if (x < 0)
		f /= pow2(-x);
	else if (x < DBL_MAX_EXP)
		f *= pow2(x);
	else
		f = (f * pow2(DBL_MAX_EXP - 1)) * pow2(x - (DBL_MAX_EXP - 1));
	return f;
}

#endif

#endif
