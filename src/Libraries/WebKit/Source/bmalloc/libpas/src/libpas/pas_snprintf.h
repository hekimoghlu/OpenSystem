/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
#ifndef PAS_SNPRINTF_H
#define PAS_SNPRINTF_H

#include "pas_utils.h"
#include <stdio.h>

PAS_BEGIN_EXTERN_C;

/* This thing is here because in some contexts, we need our own snprintf, because the available one
   calles malloc in weird cases. */

/* Returns the number of bytes *not including* the terminator that would have been written.
   The size argument is the size of the buffer *including* the terminator. */

#define pas_snprintf snprintf
#define pas_vsnprintf vsnprintf

PAS_END_EXTERN_C;

#endif /* PAS_SNPRINTF_H */

