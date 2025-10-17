/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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

#include "pas_versioned_field.h"

uintptr_t pas_versioned_field_minimize(pas_versioned_field* field,
                                       uintptr_t new_value)
{
    for (;;) {
        pas_versioned_field old_value;
        
        old_value = pas_versioned_field_read(field);
        
        if (pas_versioned_field_try_write(field,
                                          old_value,
                                          PAS_MIN(new_value, old_value.value)))
            return old_value.value;
    }
}

uintptr_t pas_versioned_field_maximize(pas_versioned_field* field,
                                       uintptr_t new_value)
{
    for (;;) {
        pas_versioned_field old_value;
        
        old_value = pas_versioned_field_read(field);
        
        if (pas_versioned_field_try_write(field,
                                          old_value,
                                          PAS_MAX(new_value, old_value.value)))
            return old_value.value;
    }
}

void pas_versioned_field_minimize_watched(pas_versioned_field* field,
                                          pas_versioned_field expected_value,
                                          uintptr_t new_value)
{
    if (new_value < expected_value.value)
        pas_versioned_field_try_write_watched(field, expected_value, new_value);
}

void pas_versioned_field_maximize_watched(pas_versioned_field* field,
                                          pas_versioned_field expected_value,
                                          uintptr_t new_value)
{
    if (new_value > expected_value.value)
        pas_versioned_field_try_write_watched(field, expected_value, new_value);
}

#endif /* LIBPAS_ENABLED */
