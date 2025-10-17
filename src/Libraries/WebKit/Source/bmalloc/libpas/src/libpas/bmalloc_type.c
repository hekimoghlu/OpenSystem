/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "bmalloc_type.h"

#include "pas_immortal_heap.h"
#include "pas_stream.h"
#include <string.h>

bmalloc_type* bmalloc_type_create(size_t size, size_t alignment, const char* name)
{
    bmalloc_type* result;

    PAS_ASSERT((unsigned)size == size);
    PAS_ASSERT((unsigned)alignment == alignment);

    result = pas_immortal_heap_allocate(
        sizeof(bmalloc_type),
        "bmalloc_type",
        pas_object_allocation);

    *result = BMALLOC_TYPE_INITIALIZER((unsigned)size, (unsigned)alignment, name);

    return result;
}

bool bmalloc_type_try_name_dump(pas_stream* stream, const char* name)
{
    const char* type_name_start_marker = "[LibPasBmallocHeapType = ";
    
    char* type_name_start_marker_ptr;
    char* type_name_start_ptr;
    unsigned bracket_balance;
    size_t index;

    type_name_start_marker_ptr = strstr(name, type_name_start_marker);
    if (!type_name_start_marker_ptr)
        return false;

    type_name_start_ptr = type_name_start_marker_ptr + strlen(type_name_start_marker);

    bracket_balance = 0;

    for (index = 0; type_name_start_ptr[index]; ++index) {
        switch (type_name_start_ptr[index]) {
        case '[':
            bracket_balance++;
            break;
        case ']':
            if (!bracket_balance) {
                char* flexible_array_member_marker;

                flexible_array_member_marker = strstr(name, "primitiveHeapRefForTypeWithFlexibleArrayMember");
                if (flexible_array_member_marker)
                    pas_stream_printf(stream, "ObjectWithFlexibleArrayMember, ");

                PAS_ASSERT((size_t)(int)index == index);
                pas_stream_printf(stream, "%.*s", (int)index, type_name_start_ptr);
                return true;
            }
            bracket_balance--;
            break;
        default:
            break;
        }
    }

    return false;
}

void bmalloc_type_name_dump(pas_stream* stream, const char* name)
{
    if (bmalloc_type_try_name_dump(stream, name))
        return;

    pas_stream_printf(stream, "%s", name);
}

void bmalloc_type_dump(const bmalloc_type* type, pas_stream* stream)
{
    pas_stream_printf(
        stream, "Size = %zu, Alignment = %zu, Type = ",
        bmalloc_type_size(type), bmalloc_type_alignment(type));

    bmalloc_type_name_dump(stream, bmalloc_type_name(type));
}

void bmalloc_type_as_heap_type_dump(const pas_heap_type* type, pas_stream* stream)
{
    bmalloc_type_dump((const bmalloc_type*)type, stream);
}

#endif /* LIBPAS_ENABLED */

