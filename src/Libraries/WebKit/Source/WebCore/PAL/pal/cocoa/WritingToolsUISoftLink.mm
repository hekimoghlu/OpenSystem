/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#import "config.h"

#if ENABLE(WRITING_TOOLS) && PLATFORM(MAC)

#import <pal/spi/cocoa/WritingToolsUISPI.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, PAL_EXPORT)

SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, WTWritingTools, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, WTWritingToolsViewController, PAL_EXPORT)

SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, _WTTextEffectView, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, _WTSweepTextEffect, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, _WTReplaceTextEffect, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, _WTReplaceSourceTextEffect, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, _WTReplaceDestinationTextEffect, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, _WTTextChunk, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, WritingToolsUI, _WTTextPreview, PAL_EXPORT)

#endif // ENABLE(WRITING_TOOLS) && PLATFORM(MAC)
