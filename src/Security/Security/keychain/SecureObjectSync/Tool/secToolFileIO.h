/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

//
//  secToolFileIO.h
//  sec
//
//
//

#ifndef secToolFileIO_h
#define secToolFileIO_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/utsname.h>
#include <time.h>
#include <CoreFoundation/CoreFoundation.h>

#define printmsg(format, ...) _printcfmsg(outFile, NULL, format, ##__VA_ARGS__)
#define printmsgWithFormatOptions(formatOptions, format, ...) _printcfmsg(outFile, formatOptions, format, ##__VA_ARGS__)
#define printerr(format, ...) _printcfmsg(errFile, NULL, format, ##__VA_ARGS__)

extern FILE *outFile;
extern FILE *errFile;

void _printcfmsg(FILE *ff, CFDictionaryRef formatOptions, CFStringRef format, ...);

int SOSLogSetOutputTo(char *dir, char *filename);

void closeOutput(void);

int copyFileToOutputDir(char *dir, char *toCopy);

#endif /* secToolFileIO_h */
