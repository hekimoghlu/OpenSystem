/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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
#ifndef	_LIBLOG_SYSTEMCONFIGURATION_INTERNAL_H
#define	_LIBLOG_SYSTEMCONFIGURATION_INTERNAL_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>
#include <os/log_private.h>
#include <os/state_private.h>

// <os/log_private.h>
#ifdef __OBJC__
#import <Foundation/NSObjCRuntime.h>
#import <Foundation/NSAttributedString.h>
typedef NSAttributedString *(*os_log_copy_formatted_fn_t)(const char *type,
							  id value, os_log_type_info_t info);
OS_EXPORT NS_RETURNS_RETAINED
NSAttributedString *
OSLogCopyFormattedString(const char *type, id value, os_log_type_info_t info);
#endif

// <os/state_private.h>
#ifdef __OBJC__
#import <Foundation/NSString.h>
typedef NSString *
(*os_state_create_string_fn_t)(const char *data_type,
			       uint32_t data_size, void *data);
OS_EXPORT NS_RETURNS_RETAINED
NSString *
OSStateCreateStringWithData(const char *data_type, uint32_t data_size, void *data);
#endif

__BEGIN_DECLS

__END_DECLS

#endif	// _LIBLOG_SYSTEMCONFIGURATION_INTERNAL_H
