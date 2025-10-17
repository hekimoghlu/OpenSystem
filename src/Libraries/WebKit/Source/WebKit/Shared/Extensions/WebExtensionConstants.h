/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

// MARK: Alarms
static constexpr auto webExtensionMinimumAlarmInterval = 30_s;

// MARK: Message Passing
/// This matches the maximum message length enforced by Chromium in its `MessageFromJSONString()` function.
static constexpr size_t webExtensionMaxMessageLength = 1024 * 1024 * 64;

// MARK: Declarative Net Request
static constexpr size_t webExtensionDeclarativeNetRequestMaximumNumberOfStaticRulesets = 100;
static constexpr size_t webExtensionDeclarativeNetRequestMaximumNumberOfEnabledRulesets = 50;
static constexpr size_t webExtensionDeclarativeNetRequestMaximumNumberOfDynamicAndSessionRules = 30000;

// MARK: Storage
static constexpr size_t webExtensionUnlimitedStorageQuotaBytes = std::numeric_limits<size_t>::max();

static constexpr size_t webExtensionStorageAreaLocalQuotaBytes = 5 * 1024 * 1024;
static constexpr size_t webExtensionStorageAreaSessionQuotaBytes = 10 * 1024 * 1024;
static constexpr size_t webExtensionStorageAreaSyncQuotaBytes = 100 * 1024;

static constexpr size_t webExtensionStorageAreaSyncQuotaBytesPerItem = 8 * 1024;

static constexpr size_t webExtensionStorageAreaSyncMaximumItems = 512;
static constexpr size_t webExtensionStorageAreaSyncMaximumWriteOperationsPerHour = 1800;
static constexpr size_t webExtensionStorageAreaSyncMaximumWriteOperationsPerMinute = 120;

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
