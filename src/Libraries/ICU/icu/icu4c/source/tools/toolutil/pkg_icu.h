/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
 *   Copyright (C) 2008-2016, International Business Machines
 *   Corporation and others.  All Rights Reserved.
 *******************************************************************************
 */

#ifndef __PKG_ICU_H__
#define __PKG_ICU_H__

#include "unicode/utypes.h"
#include "package.h"

#define U_PKG_RESERVED_CHARS "\"%&'()*+,-./:;<=>?_"

U_CAPI int U_EXPORT2
writePackageDatFile(const char *outFilename, const char *outComment,
                    const char *sourcePath, const char *addList, icu::Package *pkg,
                    char outType);

U_CAPI icu::Package * U_EXPORT2
readList(const char *filesPath, const char *listname, UBool readContents, icu::Package *listPkgIn);

#endif
