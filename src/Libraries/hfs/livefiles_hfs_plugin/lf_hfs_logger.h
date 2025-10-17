/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#ifndef lf_hfs_logger_h
#define lf_hfs_logger_h

#include <os/log.h>

typedef enum
{
    LEVEL_DEBUG,
    LEVEL_DEFAULT,
    LEVEL_ERROR,
    LEVEL_AMOUNT,

} HFSLogLevel_e;

extern os_log_t             gpsHFSLog;
extern const os_log_type_t  gpeHFSToOsLevel[ LEVEL_AMOUNT ];
extern bool                 gbIsLoggerInit;


#define LFHFS_LOG( _level, ... )                                                          \
    do {                                                                                \
        if ( gbIsLoggerInit )                                                           \
        {                                                                               \
            os_log_with_type((gpsHFSLog), gpeHFSToOsLevel[_level], ##__VA_ARGS__);  \
        }                                                                               \
    } while(0)

int LFHFS_LoggerInit( void );

#endif /* lf_hfs_logger_h */
