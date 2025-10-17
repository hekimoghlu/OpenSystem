/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
// Centralize flags which differ based on which target we are building

#ifndef DYLD_TARGET_POLICY_H
#define DYLD_TARGET_POLICY_H

// FIXME: Share this with another file
#define VIS_HIDDEN      __attribute__((visibility("hidden")))

// True if mach_o::Header adds an implicit platform to binaries if they don't have
extern VIS_HIDDEN const bool gHeaderAddImplicitPlatform;

// True if mach_o::Header is allows files to have no platform
extern VIS_HIDDEN const bool gHeaderAllowEmptyPlatform;

// True if mach_o::Image should validate initializers
extern VIS_HIDDEN const bool gImageValidateInitializers;

// True if mach_o::Image can assume content has been rebased, ie, this is dyld
extern VIS_HIDDEN const bool gImageAssumeContentRebased;

#endif /* DYLD_DEFINES_H */
