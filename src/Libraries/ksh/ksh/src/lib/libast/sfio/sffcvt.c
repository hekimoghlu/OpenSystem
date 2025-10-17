/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#include	"sfhdr.h"

#if __STD_C
char *sffcvt(double dval, int n_digit, int* decpt, int* sign)
#else
char *sffcvt(dval,n_digit,decpt,sign)
double	dval;		/* value to convert */
int	n_digit;	/* number of digits wanted */
int*	decpt;		/* to return decimal point */
int*	sign;		/* to return sign */
#endif
{
	int		len;
	static char	buf[SF_MAXDIGITS];

	return _sfcvt(&dval,buf,sizeof(buf),n_digit,decpt,sign,&len,0);
}
