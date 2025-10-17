/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
/* $Id: stdlib.h,v 1.8 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_STDLIB_H
#define ISC_STDLIB_H 1

/*! \file isc/stdlib.h */

#include <stdlib.h>

#include <isc/lang.h>
#include <isc/platform.h>

#ifdef ISC_PLATFORM_NEEDSTRTOUL
#define strtoul isc_strtoul
#endif

ISC_LANG_BEGINDECLS

unsigned long isc_strtoul(const char *, char **, int);

ISC_LANG_ENDDECLS

#endif
