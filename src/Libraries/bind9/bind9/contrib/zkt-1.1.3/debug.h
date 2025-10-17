/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#ifndef DEBUG_H
# define DEBUG_H

# ifdef DBG
#  define	dbg_line()	fprintf (stderr, "DBG: %s(%d) reached\n", __FILE__, __LINE__)
#  define	dbg_msg(msg)	fprintf (stderr, "DBG: %s(%d) %s\n", __FILE__, __LINE__, msg)
#  define	dbg_val0(text)	fprintf (stderr, "DBG: %s(%d) %s", __FILE__, __LINE__, text)
#  define	dbg_val1(fmt, var)	dbg_val (fmt, var)
#  define	dbg_val(fmt, var)	fprintf (stderr, "DBG: %s(%d) " fmt, __FILE__, __LINE__, var)
#  define	dbg_val2(fmt, v1, v2)	fprintf (stderr, "DBG: %s(%d) " fmt, __FILE__, __LINE__, v1, v2)
#  define	dbg_val3(fmt, v1, v2, v3)	fprintf (stderr, "DBG: %s(%d) " fmt, __FILE__, __LINE__, v1, v2, v3)
#  define	dbg_val4(fmt, v1, v2, v3, v4)	fprintf (stderr, "DBG: %s(%d) " fmt, __FILE__, __LINE__, v1, v2, v3, v4)
#  define	dbg_val5(fmt, v1, v2, v3, v4, v5)	fprintf (stderr, "DBG: %s(%d) " fmt, __FILE__, __LINE__, v1, v2, v3, v4, v5)
#  define	dbg_val6(fmt, v1, v2, v3, v4, v5, v6)	fprintf (stderr, "DBG: %s(%d) " fmt, __FILE__, __LINE__, v1, v2, v3, v4, v5, v6)
# else
#  define	dbg_line()
#  define	dbg_msg(msg)
#  define	dbg_val0(text)
#  define	dbg_val1(fmt, var)
#  define	dbg_val(fmt, str)
#  define	dbg_val2(fmt, v1, v2)
#  define	dbg_val3(fmt, v1, v2, v3)
#  define	dbg_val4(fmt, v1, v2, v3, v4)
#  define	dbg_val5(fmt, v1, v2, v3, v4, v5)
#  define	dbg_val6(fmt, v1, v2, v3, v4, v5, v6)
# endif

#endif
