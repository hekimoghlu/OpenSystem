/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#ifndef __CONFIGURATION_PROFILE_H__
#define __CONFIGURATION_PROFILE_H__

#include <xpc/xpc.h>
#include <os/availability.h>

char *configuration_profile_create_notification_key(const char *ident) API_AVAILABLE(macos(10.9), ios(7.0));
xpc_object_t configuration_profile_copy_property_list(const char *ident) API_AVAILABLE(macos(10.9), ios(7.0));

#endif /* __CONFIGURATION_PROFILE_H__ */
