/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
 *  BLIsNewWorld.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Thu Apr 19 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLIsNewWorld.c,v 1.9 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include <sys/param.h>	// for sysctl
#include <sys/sysctl.h>	// for sysctl

#include "bless.h"
#include "bless_private.h"

int BLIsNewWorld(BLContextPtr context)
{
    int mib[2], nw;
    size_t len;

    // do the lookup
    mib[0] = CTL_HW;
    mib[1] = HW_EPOCH;
    len = sizeof(nw);
    sysctl(mib, 2, &nw, &len, NULL, 0);

    return nw;
}
