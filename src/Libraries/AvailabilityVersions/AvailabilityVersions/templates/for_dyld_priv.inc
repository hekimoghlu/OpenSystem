/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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



// @@DYLD_HEADER_VERSIONS()@@

// These exist to maintain source compat as we cleanup headers
// TODO: These should be marked for macro deprecation and eventually removed
#define DYLD_MACOSX_VERSION_11_00               DYLD_MACOSX_VERSION_11_0
#define DYLD_MACOSX_VERSION_12_00               DYLD_MACOSX_VERSION_12_0
#define dyld_platform_version_macOS_11_00       dyld_platform_version_macOS_11_0
#define dyld_platform_version_macOS_12_00       dyld_platform_version_macOS_12_0




#if 0
#pragma clang deprecated(DYLD_MACOSX_VERSION_11_00, "use DYLD_MACOSX_VERSION_11_0 instead")
#pragma clang deprecated(DYLD_MACOSX_VERSION_12_00, "use DYLD_MACOSX_VERSION_12_0 instead")
#pragma clang deprecated(dyld_platform_version_macOS_11_00, "use dyld_platform_version_macOS_11_0 instead")
#pragma clang deprecated(dyld_platform_version_macOS_12_00, "use dyld_platform_version_macOS_12_0 instead")
#endif
