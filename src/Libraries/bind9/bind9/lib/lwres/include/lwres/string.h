/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#ifndef LWRES_STRING_H
#define LWRES_STRING_H 1

/*! \file lwres/string.h */

#include <stdlib.h>

#include <lwres/lang.h>
#include <lwres/platform.h>

#ifdef LWRES_PLATFORM_NEEDSTRLCPY
#define strlcpy lwres_strlcpy
#endif

LWRES_LANG_BEGINDECLS

size_t lwres_strlcpy(char *dst, const char *src, size_t size);

LWRES_LANG_ENDDECLS

#endif
