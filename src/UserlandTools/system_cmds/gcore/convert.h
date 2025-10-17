/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#include "options.h"

#include <stdbool.h>

#ifndef _CONVERT_H
#define _CONVERT_H

#ifdef CONFIG_GCORE_FREF
extern int gcore_fref(int);
#endif

#ifdef CONFIG_GCORE_MAP
extern int gcore_map(int);
#endif

#ifdef CONFIG_GCORE_CONV
extern int gcore_conv(int, const char *, bool, int);
#endif

#endif /* _CONVERT_H */
