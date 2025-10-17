/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#include "lf_hfs_logger.h"

os_log_t gpsHFSLog;
bool     gbIsLoggerInit = false;

const os_log_type_t gpeHFSToOsLevel [LEVEL_AMOUNT] = {
    [ LEVEL_DEBUG   ]       = OS_LOG_TYPE_DEBUG,
    [ LEVEL_DEFAULT ]       = OS_LOG_TYPE_DEFAULT,
    [ LEVEL_ERROR   ]       = OS_LOG_TYPE_ERROR
};

int
LFHFS_LoggerInit( void )
{
    int iErr = 0;
    if ( (gpsHFSLog = os_log_create("com.apple.filesystems.livefiles_hfs_plugin", "plugin")) == NULL) {
        iErr = 1;
    }
    else
    {
        gbIsLoggerInit = true;
    }

    return iErr;
}

