/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
/*
 * This header is the first thing included in any of the libarchive_fe
 * source files.  As far as possible, platform-specific issues should
 * be dealt with here and not within individual source files.
 */

#ifndef LAFE_PLATFORM_H_INCLUDED
#define	LAFE_PLATFORM_H_INCLUDED

#if defined(PLATFORM_CONFIG_H)
/* Use hand-built config.h in environments that need it. */
#include PLATFORM_CONFIG_H
#else
/* Read config.h or die trying. */
#include "config.h"
#endif

#endif
