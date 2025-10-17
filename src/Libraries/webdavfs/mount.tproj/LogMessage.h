/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#ifndef __LOGMESSAGE__
#define __LOGMESSAGE__
#ifdef __cplusplus
extern "C" {
#endif
	
#include "webdavd.h"
	
extern void logDebugCFString(const char *msg, CFStringRef str);
extern void logDebugCFURL(const char *msg, CFURLRef url);

#define LOGMESSAGEON 1

extern void LogMessage(u_int32_t level, char *format, ...);
extern u_int32_t gPrintLevel;

enum {
    kNone		= 	0x1,
    kError		= 	0x2,
    kWarning 	= 	0x4,
    kTrace		= 	0x8,
    kInfo		=	0x10,
    kTrace2		=	0x20,
    kSysLog		=	0x40,
    kAll		=	-1
};

#ifdef  __cplusplus
}
#endif

#endif /*__LOGMESSAGE__*/
