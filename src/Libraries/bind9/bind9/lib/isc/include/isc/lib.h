/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
/* $Id: lib.h,v 1.16 2009/09/02 23:48:03 tbox Exp $ */

#ifndef ISC_LIB_H
#define ISC_LIB_H 1

/*! \file isc/lib.h */

#include <isc/types.h>
#include <isc/lang.h>

ISC_LANG_BEGINDECLS

LIBISC_EXTERNAL_DATA extern isc_msgcat_t *isc_msgcat;

void
isc_lib_initmsgcat(void);
/*!<
 * \brief Initialize the ISC library's message catalog, isc_msgcat, if it
 * has not already been initialized.
 */

void
isc_lib_register(void);
/*!<
 * \brief Register the ISC library implementations for some base services
 * such as memory or event management and handling socket or timer events.
 * An external application that wants to use the ISC library must call this
 * function very early in main().
 */

ISC_LANG_ENDDECLS

#endif /* ISC_LIB_H */
