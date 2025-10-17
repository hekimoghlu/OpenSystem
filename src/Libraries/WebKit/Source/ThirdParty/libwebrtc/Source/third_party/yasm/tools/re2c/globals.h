/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

#ifndef	re2c_globals_h
#define	re2c_globals_h

#include "tools/re2c/basics.h"

extern const char *fileName;
extern char *outputFileName;
extern int sFlag;
extern int bFlag;
extern int dFlag;
extern int iFlag;
extern int bUsedYYAccept;
extern unsigned int oline;
extern unsigned int maxFill;
extern int vFillIndexes;
extern unsigned char *vUsedLabels;
extern unsigned int vUsedLabelAlloc;

extern unsigned char asc2ebc[256];
extern unsigned char ebc2asc[256];

extern unsigned char *xlat, *talx;

char *mystrdup(const char *str);

#endif
