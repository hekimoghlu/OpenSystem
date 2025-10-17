/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#ifndef _EMMTYPE_H_
#define _EMMTYPE_H_  1

#include <Security/cssmtype.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#define CSSM_HINT_CALLBACK (1)

typedef uint32 CSSM_MANAGER_EVENT_TYPES;
#define CSSM_MANAGER_SERVICE_REQUEST 1
#define CSSM_MANAGER_REPLY 2

typedef struct DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER cssm_manager_event_notification {
    CSSM_SERVICE_MASK DestinationModuleManagerType;
    CSSM_SERVICE_MASK SourceModuleManagerType;
    CSSM_MANAGER_EVENT_TYPES Event;
    uint32 EventId;
    CSSM_DATA EventData;
} CSSM_MANAGER_EVENT_NOTIFICATION DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER, *CSSM_MANAGER_EVENT_NOTIFICATION_PTR DEPRECATED_IN_MAC_OS_X_VERSION_10_7_AND_LATER;

#pragma clang diagnostic pop

#ifdef __cplusplus
}
#endif

#endif /* _EMMTYPE_H_ */
