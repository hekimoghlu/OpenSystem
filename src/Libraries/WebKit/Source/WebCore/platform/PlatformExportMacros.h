/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

#include <wtf/ExportMacros.h>

#if !defined(WEBCORE_EXPORT)

#if defined(BUILDING_WebCore) || defined(STATICALLY_LINKED_WITH_WebCore)
#define WEBCORE_EXPORT WTF_EXPORT_DECLARATION
#else
#define WEBCORE_EXPORT WTF_IMPORT_DECLARATION
#endif

#endif

#if !defined(WEBCORE_TESTSUPPORT_EXPORT)

#if defined(BUILDING_WebCoreTestSupport) || defined(STATICALLY_LINKED_WITH_WebCoreTestSupport)
#define WEBCORE_TESTSUPPORT_EXPORT WTF_EXPORT_DECLARATION
#else
#define WEBCORE_TESTSUPPORT_EXPORT WTF_IMPORT_DECLARATION
#endif

#endif
