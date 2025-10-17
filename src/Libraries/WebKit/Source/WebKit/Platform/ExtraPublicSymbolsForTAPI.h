/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#include <wtf/Platform.h>

#if PLATFORM(IOS) || PLATFORM(IOS_SIMULATOR)

// FIXME: Remove these after <rdar://problem/30772200> is fixed.
#define DECLARE_INSTALL_NAME(major, minor) \
extern __attribute__((visibility("default"))) const char install_name_ ##major## _ ##minor __asm("$ld$install_name$os" #major "." #minor "$/System/Library/PrivateFrameworks/WebKit.framework/WebKit");

DECLARE_INSTALL_NAME(4, 3);
DECLARE_INSTALL_NAME(5, 0);
DECLARE_INSTALL_NAME(5, 1);
DECLARE_INSTALL_NAME(6, 0);
DECLARE_INSTALL_NAME(6, 1);
DECLARE_INSTALL_NAME(7, 0);
DECLARE_INSTALL_NAME(7, 1);

#endif
