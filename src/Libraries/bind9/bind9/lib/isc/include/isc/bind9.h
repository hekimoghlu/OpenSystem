/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
/* $Id: bind9.h,v 1.2 2009/12/05 23:31:41 each Exp $ */

#ifndef ISC_BIND9_H
#define ISC_BIND9_H 1

#include <isc/boolean.h>
#include <isc/platform.h>

/*
 * This determines whether we are using the libisc/libdns libraries
 * in BIND9 or in some other application.  For BIND9 (named and related
 * tools) it must be set to ISC_TRUE at runtime.  Export library clients
 * will call isc_lib_register(), which will set it to ISC_FALSE.
 */
LIBISC_EXTERNAL_DATA extern isc_boolean_t isc_bind9;

#endif /* ISC_BIND9_H */
