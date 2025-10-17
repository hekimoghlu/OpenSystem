/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#import "config.h"

#if HAVE(APP_SSO)

#import <dispatch/dispatch.h>
#import <wtf/SoftLinking.h>

typedef NSString *NSErrorDomain;

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, AppSSO, PAL_EXPORT);

#if PLATFORM(MAC)
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL_WITH_EXPORT(PAL, AppSSO, SOAuthorization, PAL_EXPORT);
#else
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, AppSSO, SOAuthorization, PAL_EXPORT);
#endif

SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, AppSSO, SOAuthorizationHints, PAL_EXPORT);

SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, AppSSO, SOAuthorizationOptionUserActionInitiated, NSString*, PAL_EXPORT);

SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, AppSSO, SOAuthorizationOptionInitiatorOrigin, NSString*, PAL_EXPORT);

SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, AppSSO, SOAuthorizationOptionInitiatingAction, NSString*, PAL_EXPORT);

SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, AppSSO, SOErrorDomain, NSErrorDomain, PAL_EXPORT);

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, AppSSOCore, PAL_EXPORT);

SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, AppSSOCore, SOAuthorizationHintsCore, PAL_EXPORT);

#endif
