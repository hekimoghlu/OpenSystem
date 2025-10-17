/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
 * IPConfigurationLog.h
 * - logging related functions
 */

#ifndef _S_IPCONFIGURATIONLOG_H
#define _S_IPCONFIGURATIONLOG_H

/* 
 * Modification History
 *
 * March 25, 2013		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#include <os/log.h>
#include <sys/syslog.h>

#if NO_SYSTEMCONFIGURATION
static inline os_log_type_t
__level_to_type(int level)
{
    os_log_type_t	t;
    if (level < 0) {
	level = ~level;
    }

    switch (level) {
    case LOG_DEBUG :
	t = OS_LOG_TYPE_DEBUG;
	break;
    case LOG_EMERG:
    case LOG_ALERT:
    case LOG_CRIT:
	t = OS_LOG_TYPE_ERROR;
	break;
    case LOG_INFO:
	t = OS_LOG_TYPE_INFO;
	break;
    default:
	t = OS_LOG_TYPE_DEFAULT;
	break;
    }
    return (t);
};

#define	SC_log(__level, __format, ...)					\
    do {								\
	os_log_type_t	__type;						\
									\
	__type = __level_to_type(__level);				\
	os_log_with_type(IPConfigLogGetHandle(), __type,		\
			 __format, ## __VA_ARGS__);			\
    } while (0)
#else /* NO_SYSTEMCONFIGURATION */

#ifndef SC_LOG_HANDLE
#define SC_LOG_HANDLE	IPConfigLogGetHandle
#endif

#include <SystemConfiguration/SCPrivate.h>
#endif /* !NO_SYSTEMCONFIGURATION */

#define kIPConfigurationLogSubsystem	"com.apple.IPConfiguration"

#define kIPConfigurationLogCategoryServer	"Server"
#define kIPConfigurationLogCategoryLibrary	"Library"
#define kIPConfigurationLogCategoryHelper	"Helper"

void
IPConfigLogSetHandle(os_log_t handle);

os_log_t
IPConfigLogGetHandle(void);

#define IPConfigLog	SC_log
#define IPConfigLogFL	SC_log

#endif /* _S_IPCONFIGURATIONLOG_H */
