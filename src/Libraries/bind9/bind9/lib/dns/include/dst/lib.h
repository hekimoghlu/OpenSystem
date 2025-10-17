/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
/* $Id: lib.h,v 1.7 2007/06/19 23:47:17 tbox Exp $ */

#ifndef DST_LIB_H
#define DST_LIB_H 1

/*! \file dst/lib.h */

#include <isc/types.h>
#include <isc/lang.h>

ISC_LANG_BEGINDECLS

LIBDNS_EXTERNAL_DATA extern isc_msgcat_t *dst_msgcat;

void
dst_lib_initmsgcat(void);
/*
 * Initialize the DST library's message catalog, dst_msgcat, if it
 * has not already been initialized.
 */

ISC_LANG_ENDDECLS

#endif /* DST_LIB_H */
