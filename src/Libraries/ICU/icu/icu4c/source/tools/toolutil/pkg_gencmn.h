/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
/******************************************************************************
 *   Copyright (C) 2008, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *******************************************************************************
 */

#ifndef __PKG_GENCMN_H__
#define __PKG_GENCMN_H__

#include "unicode/utypes.h"

U_CAPI void U_EXPORT2
createCommonDataFile(const char *destDir, const char *name, const char *entrypointName, const char *type, const char *source, const char *copyRight,
                     const char *dataFile, uint32_t max_size, UBool sourceTOC, UBool verbose, char *gencmnFileName);

#endif
