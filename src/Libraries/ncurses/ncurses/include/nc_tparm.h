/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
 *  Author: Thomas E. Dickey                        2006                    *
 ****************************************************************************/

/* $Id: nc_tparm.h,v 1.6 2012/02/18 21:34:42 tom Exp $ */

#ifndef NC_TPARM_included
#define NC_TPARM_included 1

/*
 * Cast parameters past the formatting-string for tparm() to match the
 * assumption of the varargs code.
 */
#ifndef TPARM_ARG
#ifdef NCURSES_TPARM_ARG
#define TPARM_ARG NCURSES_TPARM_ARG
#else
#define TPARM_ARG long
#endif
#endif /* TPARAM_ARG */

#define TPARM_N(n) (TPARM_ARG)(n)

#define TPARM_9(a,b,c,d,e,f,g,h,i,j) tparm(a,TPARM_N(b),TPARM_N(c),TPARM_N(d),TPARM_N(e),TPARM_N(f),TPARM_N(g),TPARM_N(h),TPARM_N(i),TPARM_N(j))

#if NCURSES_TPARM_VARARGS
#define TPARM_8(a,b,c,d,e,f,g,h,i) tparm(a,TPARM_N(b),TPARM_N(c),TPARM_N(d),TPARM_N(e),TPARM_N(f),TPARM_N(g),TPARM_N(h),TPARM_N(i))
#define TPARM_7(a,b,c,d,e,f,g,h) tparm(a,TPARM_N(b),TPARM_N(c),TPARM_N(d),TPARM_N(e),TPARM_N(f),TPARM_N(g),TPARM_N(h))
#define TPARM_6(a,b,c,d,e,f,g) tparm(a,TPARM_N(b),TPARM_N(c),TPARM_N(d),TPARM_N(e),TPARM_N(f),TPARM_N(g))
#define TPARM_5(a,b,c,d,e,f) tparm(a,TPARM_N(b),TPARM_N(c),TPARM_N(d),TPARM_N(e),TPARM_N(f))
#define TPARM_4(a,b,c,d,e) tparm(a,TPARM_N(b),TPARM_N(c),TPARM_N(d),TPARM_N(e))
#define TPARM_3(a,b,c,d) tparm(a,TPARM_N(b),TPARM_N(c),TPARM_N(d))
#define TPARM_2(a,b,c) tparm(a,TPARM_N(b),TPARM_N(c))
#define TPARM_1(a,b) tparm(a,TPARM_N(b))
#define TPARM_0(a) tparm(a)
#else
#define TPARM_8(a,b,c,d,e,f,g,h,i) TPARM_9(a,b,c,d,e,f,g,h,i,0)
#define TPARM_7(a,b,c,d,e,f,g,h) TPARM_8(a,b,c,d,e,f,g,h,0)
#define TPARM_6(a,b,c,d,e,f,g) TPARM_7(a,b,c,d,e,f,g,0)
#define TPARM_5(a,b,c,d,e,f) TPARM_6(a,b,c,d,e,f,0)
#define TPARM_4(a,b,c,d,e) TPARM_5(a,b,c,d,e,0)
#define TPARM_3(a,b,c,d) TPARM_4(a,b,c,d,0)
#define TPARM_2(a,b,c) TPARM_3(a,b,c,0)
#define TPARM_1(a,b) TPARM_2(a,b,0)
#define TPARM_1(a,b) TPARM_2(a,b,0)
#define TPARM_0(a) TPARM_1(a,0)
#endif

#define TIPARM_0(s) _nc_tiparm(0,s)
#define TIPARM_1(s,a) _nc_tiparm(1,s,a)
#define TIPARM_2(s,a,b) _nc_tiparm(2,s,a,b)
#define TIPARM_3(s,a,b,c) _nc_tiparm(3,s,a,b,c)
#define TIPARM_4(s,a,b,c,d) _nc_tiparm(4,s,a,b,c,d)
#define TIPARM_5(s,a,b,c,d,e) _nc_tiparm(5,s,a,b,c,d,e)
#define TIPARM_6(s,a,b,c,d,e,f) _nc_tiparm(6,s,a,b,c,d,e,f)
#define TIPARM_7(s,a,b,c,d,e,f,g) _nc_tiparm(7,s,a,b,c,d,e,f,g)
#define TIPARM_8(s,a,b,c,d,e,f,g,h) _nc_tiparm(8,s,a,b,c,d,e,f,g,h)
#define TIPARM_9(s,a,b,c,d,e,f,g,h,i) _nc_tiparm(9,s,a,b,c,d,e,f,g,h,i)

#endif /* NC_TPARM_included */
