/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#ifndef _SCNETWORKCATEGORYTYPES_H
#define _SCNETWORKCATEGORYTYPES_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <unistd.h>
#include <CoreFoundation/CoreFoundation.h>

/*!
	@header SCNetworkCategoryTypes
 */

__BEGIN_DECLS

extern const CFStringRef 	kSCNetworkCategoryWiFiSSID
	API_AVAILABLE(macos(14.0), ios(17.0));

__END_DECLS

#endif	/* _SCNETWORKCATEGORYTYPES_H */
