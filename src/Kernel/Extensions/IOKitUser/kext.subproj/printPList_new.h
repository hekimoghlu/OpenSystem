/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
#ifndef __PRINTPLIST_NEW_H__
#define __PRINTPLIST_NEW_H__

#include <CoreFoundation/CoreFoundation.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

typedef UInt32 PListStyle;
#define kPListStyleClassic      (1)
#define kPListStyleDiagnostics  (2)

void printPList_new(FILE * stream, CFPropertyListRef plist, PListStyle style);
void showPList_new(CFPropertyListRef plist, PListStyle style);
CFMutableStringRef createCFStringForPlist_new(CFTypeRef plist, PListStyle style);

__END_DECLS

#endif /* __PRINTPLIST_NEW_H__ */
