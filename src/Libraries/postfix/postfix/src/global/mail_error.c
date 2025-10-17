/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

/* Global library. */

#include "mail_error.h"

 /*
  * The table that maps names to error bit masks. This will work on most UNIX
  * compilation environments.
  * 
  * In a some environments the table will not be linked in unless this module
  * also contains a function that is being called explicitly. REF/DEF and all
  * that.
  */
const NAME_MASK mail_error_masks[] = {
    "bounce", MAIL_ERROR_BOUNCE,
    "2bounce", MAIL_ERROR_2BOUNCE,
    "data", MAIL_ERROR_DATA,
    "delay", MAIL_ERROR_DELAY,
    "policy", MAIL_ERROR_POLICY,
    "protocol", MAIL_ERROR_PROTOCOL,
    "resource", MAIL_ERROR_RESOURCE,
    "software", MAIL_ERROR_SOFTWARE,
    0, 0,
};
