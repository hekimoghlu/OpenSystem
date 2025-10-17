/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#if defined(HAVE_CONFIG_H) && HAVE_CONFIG_H && defined(BUILDING_WITH_CMAKE)
#include "cmakeconfig.h"
#endif

#define JSC_API_AVAILABLE(...)
#define JSC_CLASS_AVAILABLE(...) JS_EXPORT
#define JSC_API_DEPRECATED(...)
#define JSC_API_DEPRECATED_WITH_REPLACEMENT(...)
// Use zero since it will be less than any possible version number.
#define JSC_MAC_VERSION_TBA 0
#define JSC_IOS_VERSION_TBA 0

#include "JSExportMacros.h"

#ifdef __cplusplus
#undef new
#undef delete
#include <wtf/FastMalloc.h>
#include <wtf/TZoneMalloc.h>
#endif
