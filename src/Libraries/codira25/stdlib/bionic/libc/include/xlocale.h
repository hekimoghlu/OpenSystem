/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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
#pragma once

/**
 * @file xlocale.h
 * @brief `locale_t` definition.
 *
 * Most users will want `<locale.h>` instead. `<xlocale.h>` is used by the C
 * library itself to export the `locale_t` type without exporting the
 * `<locale.h>` functions in other headers that export locale-sensitive
 * functions (such as `<string.h>`).
 */

#include <sys/cdefs.h>

/* If we just use void* in the typedef, the compiler exposes that in error messages. */
struct __locale_t;

/**
 * The `locale_t` type that represents a locale.
 */
typedef struct __locale_t* locale_t;
