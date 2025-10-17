/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
/*!
 @header affordance_featureflags.h - For functions related to default enabled feature flags used in tests
 */
// This file is mainly used to keep the default enabled feature flags related functionalities. It ensures that existing tests can use it as per requirements.

#ifndef _SECURITYD_AFFORDANCE_FEATUREFLAGS_H_
#define _SECURITYD_AFFORDANCE_FEATUREFLAGS_H_

#include <stdbool.h>

#ifdef    __cplusplus
extern "C" {
#endif

/// Indicates if change tracking is enabled for shared items. If `true`, changes
/// to shared items will be fetched and uploaded automatically.
bool KCSharingIsChangeTrackingEnabled(void);

/// Enables or disables change tracking for shared items. This is exposed for
/// testing only.
void KCSharingSetChangeTrackingEnabled(bool enabled);

/// Resets the change tracking state to the default.
void KCSharingClearChangeTrackingEnabledOverride(void);

#ifdef    __cplusplus
}
#endif

#endif // _SECURITYD_AFFORDANCE_FEATUREFLAGS_H_
