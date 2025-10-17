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
#ifndef _KEXTFIND_COMMANDS_H_
#define _KEXTFIND_COMMANDS_H_

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/kext/OSKext.h>

#include "kextfind_main.h"

void printKext(
    OSKextRef theKext,
    PathSpec relativePath,
    Boolean extra_info,
    char lineEnd);

void printKextProperty(
    OSKextRef theKext,
    CFStringRef propKey,
    char lineEnd);
void printKextMatchProperty(
    OSKextRef theKext,
    CFStringRef propKey,
    char lineEnd);
void printKextArches(
    OSKextRef theKext,
    char lineEnd,
    Boolean printLineEnd);

void printKextDependencies(
    OSKextRef theKext,
    PathSpec pathSpec,
    Boolean extra_info,
    char lineEnd);
void printKextDependents(
    OSKextRef theKext,
    PathSpec pathSpec,
    Boolean extra_info,
    char lineEnd);
void printKextPlugins(
    OSKextRef theKext,
    PathSpec pathSpec,
    Boolean extra_info,
    char lineEnd);

void printKextInfoDictionary(
    OSKextRef theKext,
    PathSpec pathSpec,
    char lineEnd);
void printKextExecutable(
    OSKextRef theKext,
    PathSpec pathSpec,
    char lineEnd);

CFStringRef copyPathForKext(
    OSKextRef theKext,
    PathSpec  pathSpec);

CFStringRef copyKextInfoDictionaryPath(
    OSKextRef theKext,
    PathSpec pathSpec);
CFStringRef copyKextExecutablePath(
    OSKextRef theKext,
    PathSpec pathSpec);

#endif /* _KEXTFIND_COMMANDS_H_ */
