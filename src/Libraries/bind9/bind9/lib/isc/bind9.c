/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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
/*! \file */

#include <config.h>

#include <isc/bind9.h>

/*
 * This determines whether we are using the libisc/libdns libraries
 * in BIND9 or in some other application. It is initialized to ISC_TRUE
 * and remains unchanged for BIND9 and related tools; export library
 * clients will run isc_lib_register(), which sets it to ISC_FALSE,
 * overriding certain BIND9 behaviors.
 */
LIBISC_EXTERNAL_DATA isc_boolean_t isc_bind9 = ISC_TRUE;
