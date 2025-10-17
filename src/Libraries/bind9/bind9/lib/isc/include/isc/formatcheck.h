/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
/* $Id: formatcheck.h,v 1.13 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_FORMATCHECK_H
#define ISC_FORMATCHECK_H 1

/*! \file isc/formatcheck.h */

/*%
 * ISC_FORMAT_PRINTF().
 *
 * \li fmt is the location of the format string parameter.
 * \li args is the location of the first argument (or 0 for no argument checking).
 * 
 * Note:
 * \li The first parameter is 1, not 0.
 */
#ifdef __GNUC__
#define ISC_FORMAT_PRINTF(fmt, args) __attribute__((__format__(__printf__, fmt, args)))
#else
#define ISC_FORMAT_PRINTF(fmt, args)
#endif

#endif /* ISC_FORMATCHECK_H */
