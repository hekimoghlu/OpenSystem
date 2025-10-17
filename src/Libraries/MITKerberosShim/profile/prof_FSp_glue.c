/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#include "prof_int.h"

#include <limits.h>

#include <CoreServices/CoreServices.h>

errcode_t KRB5_CALLCONV FSp_profile_init (const FSSpec* files, profile_t *ret_profile);

errcode_t KRB5_CALLCONV FSp_profile_init_path (const FSSpec* files, profile_t *ret_profile);

errcode_t KRB5_CALLCONV
FSp_profile_init (const FSSpec* files, profile_t *ret_profile)
{
    return memFullErr;
}

errcode_t KRB5_CALLCONV
FSp_profile_init_path (const FSSpec* files, profile_t *ret_profile)
{
    return FSp_profile_init (files, ret_profile);
}
