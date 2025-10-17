/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
/*
 *  BLIsMountAPFS.c
 *  bless
 *
 *  Copyright (c) 2001-2016 Apple Inc. All Rights Reserved.
 *
 */

#include <sys/mount.h>
#include <string.h>

#include "bless.h"
#include "bless_private.h"

int BLIsMountAPFS(BLContextPtr context, const char * mountpt, int *isAPFS) {
    struct statfs sc;
    
    int err;
    
    err = statfs(mountpt, &sc);
    if(err) {
        contextprintf(context, kBLLogLevelError,  "Could not statfs() %s\n", mountpt );
        return 1;
    }
    
    *isAPFS = ( !strcmp(sc.f_fstypename, "apfs") ? 1 : 0);
    
    return 0;
}
