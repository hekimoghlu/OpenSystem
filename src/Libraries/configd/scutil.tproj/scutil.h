/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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
 * November 9, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _SCUTIL_H
#define _SCUTIL_H

#include <sys/cdefs.h>
#include <histedit.h>

#define SC_LOG_HANDLE		_SC_LOG_DEFAULT

#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCPrivate.h>
#include <SystemConfiguration/SCValidation.h>

typedef struct {
	FILE		*fp;
	EditLine	*el;
	History		*h;
} Input, *InputRef;


extern AuthorizationRef		authorization;
extern InputRef			currentInput;
extern Boolean			doDispatch;
extern int			nesting;
extern SCPreferencesRef		ni_prefs;
extern CFRunLoopRef		notifyRl;
extern CFRunLoopSourceRef	notifyRls;
extern SCPreferencesRef		prefs;
extern char			*prefsPath;
extern SCDynamicStoreRef	store;
extern CFPropertyListRef	value;
extern CFMutableArrayRef	watchedKeys;
extern CFMutableArrayRef	watchedPatterns;


__BEGIN_DECLS

Boolean		process_line		(InputRef			src);

CFStringRef	_copyStringFromSTDIN	(CFStringRef			prompt,
					 CFStringRef			defaultValue);

Boolean		get_bool_from_string	(const char			*str,
					 Boolean			def_value,
					 Boolean			*ret_value,
					 Boolean			*use_default);

Boolean		get_rank_from_string	(const char			*str,
					 SCNetworkServicePrimaryRank	*ret_rank);

__END_DECLS

#endif /* !_SCUTIL_H */
