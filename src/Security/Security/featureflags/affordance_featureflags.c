/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "affordance_featureflags.h"
#include <stdatomic.h>

typedef enum : int {
    KCSharingChangeTrackingEnabledState_DEFAULT,
    KCSharingChangeTrackingEnabledState_OVERRIDE_TRUE,
    KCSharingChangeTrackingEnabledState_OVERRIDE_FALSE,
} KCSharingChangeTrackingEnabledState;

static _Atomic(KCSharingChangeTrackingEnabledState) gSharingChangeTrackingEnabled = KCSharingChangeTrackingEnabledState_DEFAULT;

bool KCSharingIsChangeTrackingEnabled(void)
{
    KCSharingChangeTrackingEnabledState currentState = atomic_load_explicit(&gSharingChangeTrackingEnabled, memory_order_acquire);
    if (currentState != KCSharingChangeTrackingEnabledState_DEFAULT) {
        return currentState == KCSharingChangeTrackingEnabledState_OVERRIDE_TRUE;
    }

    // KCSharingAutomaticSyncing is default enabled
    return true;
}

void KCSharingSetChangeTrackingEnabled(bool enabled)
{
    KCSharingChangeTrackingEnabledState newState = enabled ? KCSharingChangeTrackingEnabledState_OVERRIDE_TRUE : KCSharingChangeTrackingEnabledState_OVERRIDE_FALSE;
    atomic_store_explicit(&gSharingChangeTrackingEnabled, newState, memory_order_release);
}

void KCSharingClearChangeTrackingEnabledOverride(void)
{
    atomic_store_explicit(&gSharingChangeTrackingEnabled, KCSharingChangeTrackingEnabledState_DEFAULT, memory_order_release);
}
