/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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

#include <wtf/text/ASCIILiteral.h>

namespace WebKit {

namespace LaunchServicesDatabaseXPCConstants {

constexpr auto xpcUpdateLaunchServicesDatabaseMessageName = "update-launch-services-database-message"_s;
constexpr auto xpcLaunchServicesDatabaseKey = "launch-services-database"_s;
constexpr auto xpcRequestLaunchServicesDatabaseUpdateMessageName = "request-launch-services-database-update-message"_s;
constexpr auto xpcLaunchServicesDatabaseXPCEndpointNameKey = "xpc-endpoint-launch-services-database"_s;
constexpr auto xpcLaunchServicesDatabaseXPCEndpointMessageName = "xpc-endpoint-launch-services-database-message"_s;
} // namespace LaunchServicesDatabaseXPCConstants

} // namespace WebKit
