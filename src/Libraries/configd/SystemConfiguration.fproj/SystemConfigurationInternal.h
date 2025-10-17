/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef _SYSTEMCONFIGURATIONINTERNAL_H
#define _SYSTEMCONFIGURATIONINTERNAL_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>

#include "SCDynamicStoreInternal.h"

#if	!TARGET_OS_IPHONE
extern const CFStringRef	kSCEntNetAppleTalk;
extern const CFStringRef	kSCEntNetNetInfo;
extern const CFStringRef	kSCPropNetAppleTalkConfigMethod;
extern const CFStringRef	kSCPropNetAppleTalkDefaultZone;
extern const CFStringRef	kSCPropNetAppleTalkNetworkID;
extern const CFStringRef	kSCPropNetAppleTalkNodeID;
extern const CFStringRef	kSCValNetAppleTalkConfigMethodNode;
#endif	// !TARGET_OS_IPHONE

__BEGIN_DECLS

void
_SCDPluginExecInit		(void);

void
__SC_Log			(int		level,
				 CFStringRef	format_CF,
				 os_log_t	log,
				 os_log_type_t	type,
				 const char	*format,
				 ...)	CF_FORMAT_FUNCTION(2, 6) __attribute__((format(os_log, 5, 6)));

__END_DECLS

#endif	// _SYSTEMCONFIGURATIONINTERNAL_H
