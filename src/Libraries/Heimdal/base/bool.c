/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#include "baselocl.h"

struct heim_type_data _heim_bool_object = {
    HEIM_TID_BOOL,
    "bool-object",
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

heim_bool_t
heim_bool_create(int val)
{
    return heim_base_make_tagged_object(!!val, HEIM_TID_BOOL);
}

int
heim_bool_val(heim_bool_t ptr)
{
    return (int)heim_base_tagged_object_value(ptr);
}
