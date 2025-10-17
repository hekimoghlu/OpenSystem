/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
/* $Id: version.c,v 1.5 2007/06/19 23:47:16 tbox Exp $ */

#include <versions.h>

#include <bind9/version.h>

LIBBIND9_EXTERNAL_DATA const char bind9_version[] = VERSION;

LIBBIND9_EXTERNAL_DATA const unsigned int bind9_libinterface = LIBINTERFACE;
LIBBIND9_EXTERNAL_DATA const unsigned int bind9_librevision = LIBREVISION;
LIBBIND9_EXTERNAL_DATA const unsigned int bind9_libage = LIBAGE;
