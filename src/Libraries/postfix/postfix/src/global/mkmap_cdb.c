/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

#ifdef HAS_CDB

/* Utility library. */

#include <mymalloc.h>
#include <dict.h>

/* Application-specific. */

#include "mkmap.h"
#include <dict_cdb.h>

/* This is a dummy module, since CDB has all the functionality
 * built-in, as cdb creation requires one global lock anyway. */

MKMAP *mkmap_cdb_open(const char *unused_path)
{
    MKMAP  *mkmap = (MKMAP *) mymalloc(sizeof(*mkmap));
    mkmap->open = dict_cdb_open;
    mkmap->after_open = 0;
    mkmap->after_close = 0;
    return (mkmap);
}

#endif /* HAS_CDB */
