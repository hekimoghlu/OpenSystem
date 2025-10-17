/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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

#include <WebKit/WKBase.h>
#include <WebKit/WKFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT int WKXPCServiceMain(int argc, const char** argv) WK_API_AVAILABLE(macos(10.15), ios(13.0));
WK_EXPORT int WKAdAttributionDaemonMain(int argc, const char** argv) WK_API_AVAILABLE(macos(13.0), ios(16.0));
WK_EXPORT int WKWebPushDaemonMain(int argc, char** argv) WK_API_AVAILABLE(macos(13.0), ios(16.0));
WK_EXPORT int WKWebPushToolMain(int argc, char** argv) WK_API_AVAILABLE(macos(14.2), ios(17.2));

#ifdef __cplusplus
}
#endif
