/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
 * EAPLog.h
 * - function to log information for EAP-related routines
 */

/* 
 * Modification History
 *
 * December 26, 2012	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _EAP8021X_EAPLOG_H
#define _EAP8021X_EAPLOG_H

#include <SystemConfiguration/SCPrivate.h>
#include "symbol_scope.h"

typedef CF_ENUM(uint32_t, EAPLogCategory) {
	kEAPLogCategoryController	= 0,
	kEAPLogCategoryMonitor		= 1,
	kEAPLogCategoryClient		= 2,
	kEAPLogCategoryFramework	= 3,
};

os_log_t
EAPLogGetLogHandle();

void
EAPLogInit(EAPLogCategory log_category);

INLINE const char *
EAPLogFileName(const char * file)
{
    const char *	ret;

    ret = strrchr(file, '/');
    if (ret != NULL) {
	ret++;
    }
    else {
	ret = file;
    }
    return (ret);
}

#define EAPLOG(level, format, ...)											\
    do {																	\
		os_log_t log_handle = EAPLogGetLogHandle();					\
		os_log_type_t __type = _SC_syslog_os_log_mapping(level);			\
		os_log_with_type(log_handle, __type, format, ## __VA_ARGS__);		\
	} while (0)

#define EAPLOG_FL EAPLOG

#endif /* _EAP8021X_EAPLOG_H */
