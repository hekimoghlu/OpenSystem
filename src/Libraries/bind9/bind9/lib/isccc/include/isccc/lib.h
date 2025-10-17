/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
/* $Id: lib.h,v 1.11 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_LIB_H
#define ISCCC_LIB_H 1

/*! \file isccc/lib.h */

#include <isc/types.h>
#include <isc/lang.h>

ISC_LANG_BEGINDECLS

LIBISCCC_EXTERNAL_DATA extern isc_msgcat_t *isccc_msgcat;

void
isccc_lib_initmsgcat(void);
/*%
 * Initialize the ISCCC library's message catalog, isccc_msgcat, if it
 * has not already been initialized.
 */

ISC_LANG_ENDDECLS

#endif /* ISCCC_LIB_H */
