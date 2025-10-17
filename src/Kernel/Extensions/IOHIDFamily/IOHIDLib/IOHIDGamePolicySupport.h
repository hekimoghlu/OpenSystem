/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#pragma once

#include <IOKit/IOKitLib.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

typedef void(^ConsoleModeBlock)(bool status);

/*!
 * IOHIDAnalyticsGetConsoleModeStatus
 *
 * @abstract
 * Get the current app console mode state according to GamePolicy
 *
 * @discussion
 * Connects to an XPC service running in gamepolicyd to fetch the state.
 *
 * @param status
 * where the current state value is returned if kIOReturnSuccess is returned. Otherwise, the value will not change.
 *
 * @param timeout
 * The timeout in nanoseconds to wait for the response.
 * 
 * @return
 * kIOReturnSuccess if the state was fetched before the timeout, kIOReturnTimeout otherwise.
 * kIOReturnError if the framework fails to load.
 * 
 */
CF_EXPORT
IOReturn IOHIDAnalyticsGetConsoleModeStatus(ConsoleModeBlock block);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS
