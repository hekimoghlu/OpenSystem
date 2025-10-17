/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
/* 
 * Defining here VECTORS for all files that include this header (<rdar://problem/8466056>)
 */
#ifndef VECTORS
#define VECTORS
typedef __attribute__ ((vector_size(16))) unsigned char VECTORTYPE;
#ifdef __SSE2__
#define V64TYPE
#endif /* __SSE2__ */
#endif /* VECTORS */

/*
 * Flags used during conversion.
 */
#define	ALT		0x001		/* alternate form */
#define	LADJUST		0x004		/* left adjustment */
#define	LONGDBL		0x008		/* long double */
#define	LONGINT		0x010		/* long integer */
#define	LLONGINT	0x020		/* long long integer */
#define	SHORTINT	0x040		/* short integer */
#define	ZEROPAD		0x080		/* zero (as opposed to blank) pad */
#define	FPT		0x100		/* Floating point number */
#define	GROUPING	0x200		/* use grouping ("'" flag) */
					/* C99 additional size modifiers: */
#define	SIZET		0x400		/* size_t */
#define	PTRDIFFT	0x800		/* ptrdiff_t */
#define	INTMAXT		0x1000		/* intmax_t */
#define	CHARINT		0x2000		/* print char using int format */
#ifdef VECTORS
#define VECTOR          0x4000          /* Altivec or SSE vector */
#endif /* VECTORS */

/*
 * Macros for converting digits to letters and vice versa
 */
#define	to_digit(c)	((c) - '0')
#define is_digit(c)	((unsigned)to_digit(c) <= 9)
#define	to_char(n)	((n) + '0')

/* Size of the static argument table. */
#define STATIC_ARG_TBL_SIZE 8

union arg {
	int	intarg;
	u_int	uintarg;
	long	longarg;
	u_long	ulongarg;
	long long longlongarg;
	unsigned long long ulonglongarg;
	ptrdiff_t ptrdiffarg;
	size_t	sizearg;
	intmax_t intmaxarg;
	uintmax_t uintmaxarg;
	void	*pvoidarg;
	char	*pchararg;
	signed char *pschararg;
	short	*pshortarg;
	int	*pintarg;
	long	*plongarg;
	long long *plonglongarg;
	ptrdiff_t *pptrdiffarg;
	ssize_t	*pssizearg;
	intmax_t *pintmaxarg;
#ifndef NO_FLOATING_POINT
	double	doublearg;
	long double longdoublearg;
#endif
	wint_t	wintarg;
	wchar_t	*pwchararg;
#ifdef VECTORS
	VECTORTYPE		vectorarg;
	unsigned char		vuchararg[16];
	signed char		vchararg[16];
	unsigned short		vushortarg[8];
	signed short		vshortarg[8];
	unsigned int		vuintarg[4];
	signed int		vintarg[4];
	float			vfloatarg[4];
#ifdef V64TYPE
	double			vdoublearg[2];
	unsigned long long	vulonglongarg[2];
	long long		vlonglongarg[2];
#endif /* V64TYPE */
#endif /* VECTORS */
};

/* Handle positional parameters. */
int	__find_arguments(const char *, va_list, union arg **);
int	__find_warguments(const wchar_t *, va_list, union arg **);
