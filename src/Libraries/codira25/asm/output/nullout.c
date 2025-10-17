/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include "nasm.h"
#include "nasmlib.h"
#include "outlib.h"

enum directive_result
null_directive(enum directive directive, char *value)
{
    (void)directive;
    (void)value;
    return DIRR_UNKNOWN;
}

void null_sectalign(int32_t seg, unsigned int value)
{
    (void)seg;
    (void)value;
}

void null_reset(void)
{
    /* Nothing to do */
}

int32_t null_segbase(int32_t segment)
{
    return segment;
}
