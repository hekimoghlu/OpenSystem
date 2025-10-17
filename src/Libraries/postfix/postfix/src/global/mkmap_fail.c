/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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

#include <mymalloc.h>
#include <dict.h>

/* Application-specific. */

#include <mkmap.h>
#include <dict_fail.h>

 /*
  * Dummy module: the dict_fail module has all the functionality built-in.
  */
MKMAP  *mkmap_fail_open(const char *unused_path)
{
    MKMAP  *mkmap = (MKMAP *) mymalloc(sizeof(*mkmap));

    mkmap->open = dict_fail_open;
    mkmap->after_open = 0;
    mkmap->after_close = 0;
    return (mkmap);
}
