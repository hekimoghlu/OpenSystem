/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#ifndef _OBJC_FILE_NEW_H
#define _OBJC_FILE_NEW_H

#include "objc-runtime-new.h"

// classref_t is not fixed up at launch; use remapClass() to convert

// classlist and catlist and protolist sections are marked const here
// because those sections may be in read-only __DATA_CONST segments.

static inline void
foreach_data_segment(const headerType *mhdr,
                     std::function<void(const segmentType *, intptr_t slide)> code)
{
    intptr_t slide = 0;

    // compute VM slide
    const segmentType *seg = (const segmentType *) (mhdr + 1);
    for (unsigned long i = 0; i < mhdr->ncmds; i++) {
        if (seg->cmd == SEGMENT_CMD  &&
            segnameEquals(seg->segname, "__TEXT"))
        {
            slide = (char *)mhdr - (char *)seg->vmaddr;
            break;
        }
        seg = (const segmentType *)((char *)seg + seg->cmdsize);
    }

    // enumerate __DATA* and __AUTH* segments
    seg = (const segmentType *) (mhdr + 1);
    for (unsigned long i = 0; i < mhdr->ncmds; i++) {
        if (seg->cmd == SEGMENT_CMD  &&
            (segnameStartsWith(seg->segname, "__DATA") ||
             segnameStartsWith(seg->segname, "__AUTH")))
        {
            code(seg, slide);
        }
        seg = (const segmentType *)((char *)seg + seg->cmdsize);
    }
}

#endif
