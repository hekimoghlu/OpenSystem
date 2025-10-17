/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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

#include <pal/spi/cocoa/AccessibilitySupportSPI.h>
#include <wtf/SoftLinking.h>

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)

SOFT_LINK_LIBRARY_FOR_HEADER(PAL, libAccessibility)
SOFT_LINK_FUNCTION_MAY_FAIL_FOR_HEADER(PAL, libAccessibility, _AXSIsolatedTreeMode, AXSIsolatedTreeMode, (void), ())
#define _AXSIsolatedTreeMode_Soft PAL::softLink_libAccessibility__AXSIsolatedTreeMode
#define _AXSIsolatedTreeModeFunctionIsAvailable PAL::islibAccessibilityLibaryAvailable() && PAL::canLoad_libAccessibility__AXSIsolatedTreeMode

#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)
