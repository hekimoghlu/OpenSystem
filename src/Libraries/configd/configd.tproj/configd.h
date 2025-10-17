/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 17, 2023.
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
 * Modification History
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * March 24, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _S_CONFIGD_H
#define _S_CONFIGD_H

#include <stdio.h>
#include <stdlib.h>
#include <TargetConditionals.h>

/* configd doesn't need the preference keys */
#define _SCSCHEMADEFINITIONS_H
#define _SCSCHEMADEFINITIONSPRIVATE_H

#define	SC_LOG_HANDLE	__configd_SCDynamicStore
#include "SCDynamicStoreInternal.h"
#include "config_types.h"
#include "_SCD.h"

extern Boolean		_configd_verbose;		/* TRUE if verbose logging enabled */
extern CFMutableSetRef	_plugins_exclude;		/* bundle identifiers to exclude from loading */
extern CFMutableSetRef	_plugins_verbose;		/* bundle identifiers to enable verbose logging */

extern Boolean	_should_log_path;

/*
 * PrivacyAccounting framework is only available on (non-simulator):
 * - iphoneos
 * - watchos
 */
#if (TARGET_OS_IOS || TARGET_OS_WATCH) && !TARGET_OS_SIMULATOR
#define _HAVE_PRIVACY_ACCOUNTING		1
#else
#define _HAVE_PRIVACY_ACCOUNTING		0
#endif

/*
 * BASupport library is only available on
 * - iphoneos
 * - macOS
 */
#if TARGET_OS_IOS || TARGET_OS_OSX
#define _HAVE_BASUPPORT				1
#else
#define _HAVE_BASUPPORT				0
#endif

#define SC_trace(__string, ...)	\
	os_log_debug(SC_LOG_HANDLE(), __string, ## __VA_ARGS__)


__BEGIN_DECLS

os_log_t
__configd_SCDynamicStore	(void);

#if _HAVE_PRIVACY_ACCOUNTING
Boolean
isSystemProcess(CFDictionaryRef entitlements);

Boolean
havePrivacyAccounting(void);

#endif /* _HAVE_PRIVACY_ACCOUNTING */

void
ALT_CFRelease(CFTypeRef cf);

__END_DECLS

#endif	/* !_S_CONFIGD_H */
