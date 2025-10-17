/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#ifndef WKUtilities_h
#define WKUtilities_h

#if TARGET_OS_IPHONE

#import "WKTypes.h"
#import <CoreGraphics/CoreGraphics.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const CFArrayCallBacks WKCollectionArrayCallBacks;
extern const CFSetCallBacks WKCollectionSetCallBacks;


typedef void(*WKDeallocCallback)(WAKObjectRef object);

typedef struct _WKClassInfo WKClassInfo;

struct _WKClassInfo
{
    const WKClassInfo *parent;
    const char *name;
    WKDeallocCallback dealloc;
};

extern WKClassInfo WAKObjectClass;

struct _WAKObject
{
    unsigned referenceCount;
    WKClassInfo *classInfo;
};

const void *WKCreateObjectWithSize (size_t size, WKClassInfo *info);
// These functions use the WAK prefix to avoid breaking InstallAPI in Mac
// Catalyst: https://bugs.webkit.org/show_bug.cgi?id=257560
const void *WAKRetain(const void *object);
void WAKRelease(const void *object);

const void *WKCollectionRetain (CFAllocatorRef allocator, const void *value);
void WKCollectionRelease (CFAllocatorRef allocator, const void *value);

void WKReportError(const char *file, int line, const char *function, const char *format, ...);
#define WKError(formatAndArgs...) WKReportError(__FILE__, __LINE__, __PRETTY_FUNCTION__, formatAndArgs)

CFIndex WKArrayIndexOfValue (CFArrayRef array, const void *value);

WKClassInfo *WKGetClassInfo(WAKObjectRef);

#ifdef __cplusplus
}
#endif

#endif // TARGET_OS_IPHONE

#endif // WKUtilities_h
