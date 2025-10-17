/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
***************************************************************************
*   Copyright (C) 2006 International Business Machines Corporation        *
*   and others. All rights reserved.                                      *
***************************************************************************
*/

#ifndef LOCALSVC_H
#define LOCALSVC_H

#include "unicode/utypes.h"

#if defined(U_LOCAL_SERVICE_HOOK) && U_LOCAL_SERVICE_HOOK
/**
 * Prototype for user-supplied service hook. This function is expected to return
 * a type of factory object specific to the requested service.
 * 
 * @param what service-specific string identifying the specific user hook
 * @param status error status
 * @return a service-specific hook, or NULL on failure.
 */
U_CAPI void* uprv_svc_hook(const char *what, UErrorCode *status);
#endif

#endif
